import torch
import torch.nn as nn
from full_transformer import MAX_LENGTH, EMBEDDING_SIZE, HEAD_SIZE, NUM_DECODER_BLOCKS, NUM_HEADS, token_to_char, vocab_size, GPT, tokenizer
import json
import random

BATCH_SIZE = 8

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# update the tokenizer with new tokens for user message start, assistant message start, end of sequence, and padding tokens
new_tokens = ["<user>", "<assistant>", "<end>"]
for tok in new_tokens:
    if tok not in tokenizer:
        token_to_char[len(tokenizer)] = tok
        tokenizer[tok] = len(tokenizer)
tokenizer["<pad>"] = -100

new_vocab_size = len(tokenizer)

def update_embeddings(old_text_embeddings: nn.Embedding, new_vocab_size: int):
    new_text_embeddings = nn.Embedding(new_vocab_size, EMBEDDING_SIZE, device=device)
    with torch.no_grad():
        new_text_embeddings.weight[:old_text_embeddings.weight.size(0), :] = old_text_embeddings.weight
    return new_text_embeddings

def update_final_proj_layer(old_linear_proj: nn.Linear, new_vocab_size: int):
    new_linear_output = nn.Linear(EMBEDDING_SIZE, new_vocab_size, device=device)

    with torch.no_grad():
        new_linear_output.weight[:old_linear_proj.weight.size(0), :] = old_linear_proj.weight
        new_linear_output.bias[:old_linear_proj.bias.size(0)] = old_linear_proj.bias

    return new_linear_output

def tokenization_helper(input_text: str, output_text: str, tokenizer: dict):
    input_seq = []
    output_seq = []

    if "<user>" in input_text:
        input_seq.append("<user>")
        new_text = input_text.split("<user>")[1]
        if "<assistant>" in new_text:
            input_seq.append(new_text.split("<assistant>")[0])
            input_seq.append("<assistant>")
            input_seq.append(new_text.split("<assistant>")[1])

        else:
            raise Exception("No <assistant> tag in input.")
    else:
        raise Exception("No <user> tag in input.")

    if "<end>" in output_text:
        output_seq.append(output_text.split("<end>")[0])
        output_seq.append("<end>")
    else:
        raise Exception("No <user> tag in input.")

    tokenized_seq = []
    seq_to_tokenize = input_seq + output_seq
    
    assert len(seq_to_tokenize) == len(input_seq) + len(output_seq)

    for item in seq_to_tokenize:
        if item in ["<user>", "<assistant>", "<end>", "<pad>"]:
            tokenized_seq.append(tokenizer.get(item))
        else:
            tokenized_seq.extend([tokenizer.get(char) for char in item])

    batch_labels = [-100]*len(input_seq) + tokenized_seq[len(input_seq):]

    return tokenized_seq, batch_labels

def pad_batch(data, tokenizer):
    max_length_for_batch_input = max([len(d[0]) for d in data])
    max_length_for_batch_output = max([len(d[1]) for d in data])

    input_stack = []
    output_stack = []

    for d in data:
        for data_in, data_out in d:

            if len(data_in) < max_length_for_batch_input:
                len_data_in = len(data_in)
                data_in.extend([tokenizer.get("<pad>")]*(max_length_for_batch_input-len_data_in))
                input_stack.append(torch.tensor(data_in, device=torch.device("cpu")))

            if len(data_out) < max_length_for_batch_output:
                len_data_out = len(data_out)
                data_out.extend([tokenizer.get("<pad>")]*(max_length_for_batch_output-len_data_out))
                output_stack.append(torch.tensor(data_out, device=torch.device("cpu")))

    batch_input = torch.stack(input_stack)
    batch_output = torch.stack(output_stack)

    return batch_input, batch_output

def make_batches(raw_data, tokenizer):
    dataset = []
    k = 0
    while k < len(raw_data):
        batch = raw_data[k:k+BATCH_SIZE]
        batch_input_tokens, batch_output_tokens = pad_batch(data=batch, tokenizer=tokenizer)
        dataset.append([batch_input_tokens, batch_output_tokens])
        k += BATCH_SIZE
    return dataset

# define eval function for intermediate generation
def sft_interm_eval(model: GPT, interval: int, phase: str):
    model.eval()
    
    print("*"*100)
    print(f"Generating after {interval} {phase} batches: ")
    
    model.generate(500, token_to_char=token_to_char, max_context_length=MAX_LENGTH)
    model.generate_assistant_response(question="Who is Romeo?", tokenizer=tokenizer, max_tokens=500, token_to_char=token_to_char, max_context_length=MAX_LENGTH)
    
    print("\n")
    print("*"*100)
    
    model.train()

if __name__ == "__main__":
    # load model 
    model = GPT(vocab_size=vocab_size, max_context_length=MAX_LENGTH, embedding_size=EMBEDDING_SIZE, num_heads=NUM_HEADS, head_size=HEAD_SIZE, num_decoder_layers=NUM_DECODER_BLOCKS)
    model.load_state_dict(torch.load("./full_transformer_tiny_shakespeare.pt"))

    # initialize new embedding and final linear output projection layer taking into account special tokens
    model.text_embedding_table = update_embeddings(old_text_embeddings=model.text_embedding_table, new_vocab_size=new_vocab_size)
    model.linear_output = update_final_proj_layer(old_linear_proj=model.linear_output, new_vocab_size=new_vocab_size)

    # load training components
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # to ignore padded tokens
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)

    raw_dataset = []

    try:
        with open("/home/nz-dgx-spark-01/Documents/Nyalazone/pytorch_testing/datasets/sft_shakespeare_dataset/sft_dataset.jsonl", mode="r") as f:
            for line in f:
                data_entry = json.loads(s=line)
                input_tokens, output_tokens = tokenization_helper(input_text=data_entry["input"], output_text=data_entry["output"], tokenizer=tokenizer)
                print(input_tokens)
                print(output_tokens)
                exit()
                raw_dataset.append([input_tokens, output_tokens])
    except Exception as e:
        print(str(e))

    # split again 90-10 and shuffle
    train_raw = raw_dataset[:int(0.9*len(raw_dataset))]
    test_raw = raw_dataset[int(0.9*len(raw_dataset)):]

    random.shuffle(train_raw)
    random.shuffle(test_raw)

    # Making batches
    train_dataset = make_batches(raw_data=train_raw, tokenizer=tokenizer)
    test_dataset = make_batches(raw_data=test_raw, tokenizer=tokenizer)

    best_test_loss = float('inf')
    num_tries = 0

    for epoch in range(1, 100):

        random.shuffle(train_dataset)
        model.train()
        epoch_train_loss_vals = []
        train_interval = 0
        epoch_test_loss_vals = []

        # going through training batches
        for data_in, data_out in train_dataset:

            optimizer.zero_grad()

            pred = model(data_in.to(device))

            B, T, C = pred.shape

            loss = loss_fn(pred.view(B*T, C), data_out.to(device).view(B*T))

            epoch_train_loss_vals.append(loss.item())

            loss.backward()
            # TODO: Add gradient clipping later
            optimizer.step()

            if train_interval % 8000 == 0:
                sft_interm_eval(model=model, interval=train_interval, phase="Train")
            train_interval += 1

        # going through test batches
        model.eval()
        with torch.no_grad():
            for test_data_in, test_data_out in test_dataset:
                pred = model(test_data_in.to(device))
                
                B, T, C = pred.shape
                
                loss = loss_fn(pred.view(B*T, C), test_data_out.to(device).view(B*T))
                
                epoch_test_loss_vals.append(loss.item())

        avg_train_loss = sum(epoch_train_loss_vals)/float(len(epoch_train_loss_vals))
        avg_test_loss = sum(epoch_test_loss_vals)/float(len(epoch_test_loss_vals))

        print("\nEpoch ", epoch, "\nAvg Training loss: ", avg_train_loss, "\nAvg Test Loss: ", avg_test_loss, "\n")
        print("Generate: ")
        model.generate_assistant_response(question="Who is Romeo?", tokenizer=tokenizer, max_tokens=500, token_to_char=token_to_char, max_context_length=MAX_LENGTH)
        print("\n")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            print("Saving model....\n\n")
            torch.save(obj=model.state_dict(), f="./full_transformer_tiny_shakespeare_sft.pt")
            num_tries = 0
        elif num_tries > 0:
            print("Loss did not decrease for 2 times in a row, exiting....\n\n")
            break
        else:
            print("Loss greater than previous epoch, not saving model\n\n")
            num_tries += 1