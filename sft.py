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

def tokenization_helper(text: str, tokenizer: dict, text_type: str):
    seq_to_tokenize = []

    if text_type == "input":
        if "<user>" in text:
            seq_to_tokenize.append("<user>")
            new_text = text.split("<user>")[1]
            if "<assistant>" in new_text:
                seq_to_tokenize.append(new_text.split("<assistant>")[0])
                seq_to_tokenize.append("<assistant>")
                seq_to_tokenize.append(new_text.split("<assistant>")[1])

            else:
                raise Exception("No <assistant> tag in input.")
        else:
            raise Exception("No <user> tag in input.")

    if text_type == "output":
        if "<end>" in text:
            seq_to_tokenize.append(text.split("<end>")[0])
            seq_to_tokenize.append("<end>")
        else:
            raise Exception("No <user> tag in input.")

    tokenized_seq = []

    for item in seq_to_tokenize:
        if item in ["<user>", "<assistant>", "<end>", "<pad>"]:
            tokenized_seq.append(tokenizer.get(item))
        else:
            tokenized_seq.extend([tokenizer.get(char) for char in item])

    return tokenized_seq

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
                input_tokens = tokenization_helper(text=data_entry["input"], text_type="input", tokenizer=tokenizer)
                output_tokens = tokenization_helper(text=data_entry["output"], text_type="output", tokenizer=tokenizer)
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