import torch 
import torch.nn as nn
import random
import sys
import glob
import json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MAX_LENGTH = 300
BATCH_SIZE = 32
EMBEDDING_SIZE = 200

HEAD_SIZE = 20
NUM_HEADS = 10
NUM_DECODER_BLOCKS = 6
DROPOUT_VAL = 0.1
DATASET_TO_USE = "tiny_shakespeare"
# DATASET_TO_USE = "full_shakespeare"

class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.w_q = nn.Linear(embedding_size, head_size, device=device)
        self.w_k = nn.Linear(embedding_size, head_size, device=device)
        self.w_v = nn.Linear(embedding_size, head_size, device=device)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=DROPOUT_VAL)

    def forward(self, x):

        # this is after x has gotten text and position embedding and gone through layer norm
        B, C, E = x.shape # Batch, context length, embedding size

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        query_key_result = (q @ k.transpose(1, 2))/(self.head_size ** 0.5) # (B, C, head_size) x (B, head_size, C) = (batch, C, C)
        query_key_result = query_key_result.masked_fill(torch.tril(torch.ones(C, C, device=device)) == 0, float('-inf'))
        query_key_result = self.softmax(query_key_result)
        query_key_result = self.dropout(query_key_result)
        attn_out = query_key_result @ v # (B, C, C) * (B, C, head_size) = (B, C, head_size)

        return attn_out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_size, head_size):
        super().__init__()
        self.attn_heads = nn.ModuleList(
            [
                AttentionHead(embedding_size=embedding_size, head_size=head_size)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size*num_heads, embedding_size, device=device)
        self.dropout = nn.Dropout(p=DROPOUT_VAL)

    def forward(self, x):
        # after text and position embeddings and layer norm are done
        x = torch.cat([head(x) for head in self.attn_heads], dim=-1) # (B, C, head_size*num_heads)
        x = self.dropout(self.proj(x)) # (B, C, head_size*num_heads) x (head_size*num_heads, E) = (B, C, E)

        return x

class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.l1 = nn.Linear(embedding_size, 4*embedding_size, device=device) # B, C, 4*E
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4*embedding_size, embedding_size, device=device) # B, C, E
        self.dropout = nn.Dropout(p=DROPOUT_VAL)

    def forward(self, x):

        # x after passing through attention and going through layer norm
        return self.dropout(self.l2(self.relu(self.l1(x))))

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, head_size):
        super().__init__()
        self.attn_block = MultiHeadAttention(num_heads=num_heads, embedding_size=embedding_size, head_size=head_size)
        self.feed_forward_block = FeedForward(embedding_size=embedding_size)
        self.layer_norm_1 = nn.LayerNorm(embedding_size, device=device)
        self.layer_norm_2 = nn.LayerNorm(embedding_size, device=device)

    def forward(self, x):

        # after text and positional encoding
        x = x + self.attn_block(self.layer_norm_1(x)) # adding x back for residual connection (B, C, E)
        x = x + self.feed_forward_block(self.layer_norm_2(x)) # adding x back for residual connection (B, C, E)

        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, max_context_length, embedding_size, num_heads, head_size, num_decoder_layers):
        super().__init__()
        self.text_embedding_table = nn.Embedding(vocab_size, embedding_size, device=device)
        self.position_embedding_table = nn.Embedding(max_context_length, embedding_size, device=device)
        self.transformer = nn.ModuleList(
            [
                TransformerDecoderBlock(embedding_size=embedding_size, num_heads=num_heads, head_size=head_size)
                for _ in range(num_decoder_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_size, device=device)
        self.linear_output = nn.Linear(embedding_size, vocab_size, device=device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # gets the embeddings for the textual content in each batch
        text_embeddings = self.text_embedding_table(x) # B, C -> B, C, E
        # gets the embeddings for the positions of each element of each batch
        input_positions = torch.stack([torch.arange(len(item), device=device) for item in x])
        position_embeddings = self.position_embedding_table(input_positions) # B, C -> B, C, E

        x = text_embeddings + position_embeddings # B, C, E

        for decoder_block in self.transformer:
            x = decoder_block(x)

        x = self.linear_output(self.final_layer_norm(x)) # (B, C, E) x (E, vocab_size) = (B, C, vocab_size)

        return x

    def generate(self, max_tokens, token_to_char, max_context_length, start_token=1):
        with torch.no_grad():
            sequence = torch.tensor([start_token], device=device).unsqueeze(0) if type(start_token) == int else torch.tensor(start_token, device=device).unsqueeze(0)
            for _ in range(max_tokens):
                preds = self(sequence[:, -max_context_length:]) # 1, C, vocab_size - making sure only the max context length tokens go in
                preds = preds[:, -1, :] # only interested in the last token vals - 1, vocab_size
                preds = self.softmax(preds) # get probabilities of each word in vocab - 1, vocab_size
                pred = torch.multinomial(preds, num_samples=1)
                pred_token = token_to_char.get(pred.item())
                print(pred_token, end="", flush=True)

                sequence = torch.cat([sequence, pred], dim=1)

    def generate_assistant_response(self, question, tokenizer, max_tokens, token_to_char, max_context_length):
        final_input = [tokenizer.get("<user>"), tokenizer.get("\n"), *[tokenizer.get(char) for char in question], tokenizer.get("\n"), tokenizer.get("<assistant>"), tokenizer.get("\n")]
        self.generate(max_tokens=max_tokens, token_to_char=token_to_char, max_context_length=max_context_length, start_token=final_input)

if DATASET_TO_USE == "tiny_shakespeare":
    with open("/home/nz-dgx-spark-01/Documents/Nyalazone/pytorch_testing/shakespeare_dataset.txt", mode="r") as f:
        raw_data = f.read()
elif DATASET_TO_USE == "full_shakespeare":
    ROOT_DIR = "/home/nz-dgx-spark-01/Documents/Nyalazone/pytorch_testing/datasets/shakespeare-dataset/text"
    raw_data = ""

    for file_path in glob.glob("*.txt", root_dir=ROOT_DIR):
        with open(f"{ROOT_DIR}/{file_path}", mode="r") as f:
            raw_data += "\n" + f.read().replace("\u2019", "'")
else:
    print("Incorrect dataset to use.")
    sys.exit(1)

raw_unique_chars = list(sorted(set(raw_data)))

tokenizer = {char: i for i, char in enumerate(raw_unique_chars)}
token_to_char = {i: char for i, char in enumerate(raw_unique_chars)}
encode = lambda x: [tokenizer.get(c) for c in x]

vocab_size = len(raw_unique_chars)

# Training components initialized
model = GPT(vocab_size=vocab_size, max_context_length=MAX_LENGTH, embedding_size=EMBEDDING_SIZE, num_heads=NUM_HEADS, head_size=HEAD_SIZE, num_decoder_layers=NUM_DECODER_BLOCKS)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)

if __name__ == "__main__":
    j = 0
    raw_dataset = []

    while j < len(raw_data):

        # skip sequences whose length is not equal to max length (usually will be end sequences)
        if len(raw_data[j:j+MAX_LENGTH]) == MAX_LENGTH and len(raw_data[j+1:j+1+MAX_LENGTH]) == MAX_LENGTH:
            new_data = [torch.tensor(encode(raw_data[j:j+MAX_LENGTH]), device=torch.device("cpu")), torch.tensor(encode(raw_data[j+1:j+1+MAX_LENGTH]), device=torch.device("cpu"))]
            raw_dataset.append(new_data)

        j += 1

    train_raw = raw_dataset[:int(0.9*len(raw_dataset))]
    test_raw = raw_dataset[int(0.9*len(raw_dataset)):]

    # shuffle data before splitting into batches
    random.shuffle(train_raw)
    random.shuffle(test_raw)

    def make_batches(raw_data):
        dataset = []
        k = 0
        while k < len(raw_data):
            batch = raw_data[k:k+BATCH_SIZE]

            in_tokens = torch.stack([data[0] for data in batch])
            out_tokens = torch.stack([data[1] for data in batch])

            dataset.append([in_tokens, out_tokens])
            
            k += BATCH_SIZE

        return dataset

    # break into training and test sets (90-10 split)
    train_dataset = make_batches(raw_data=train_raw)
    test_dataset = make_batches(raw_data=test_raw)

    # define eval function for intermediate generation
    def interm_eval(model: GPT, interval: int, phase: str):
        model.eval()
        
        print("*"*100)
        print(f"Generating after {interval} {phase} batches: ")
        
        model.generate(500, token_to_char=token_to_char, max_context_length=MAX_LENGTH)
        
        print("\n")
        print("*"*100)
        
        model.train()

    # training loop start

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
                interm_eval(model=model, interval=train_interval, phase="Train")
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
        model.generate(500, token_to_char=token_to_char, max_context_length=MAX_LENGTH)
        print("\n")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            print("Saving model....\n\n")
            torch.save(obj=model.state_dict(), f="./full_transformer_tiny_shakespeare.pt")
            num_tries = 0
        elif num_tries > 0:
            print("Loss did not decrease for 2 times in a row, exiting....\n\n")
            break
        else:
            print("Loss greater than previous epoch, not saving model\n\n")
            num_tries += 1