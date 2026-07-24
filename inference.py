import torch
import torch.nn as nn
import json

device = torch.device("cuda")

MAX_LENGTH = 200
BATCH_SIZE = 32
EMBEDDING_SIZE = 200
HEAD_SIZE = 20
NUM_HEADS = 10
NUM_DECODER_BLOCKS = 6
vocab_size = 65

with open("/home/nz-dgx-spark-01/Documents/Nyalazone/pytorch_testing/tokenizer.json", mode="r") as f:
    tokenizer = json.load(fp=f)

token_to_char = {token: char for char, token in tokenizer.items()}

class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.w_q = nn.Linear(embedding_size, head_size, device=device)
        self.w_k = nn.Linear(embedding_size, head_size, device=device)
        self.w_v = nn.Linear(embedding_size, head_size, device=device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        # this is after x has gotten text and position embedding and gone through layer norm
        B, C, E = x.shape # Batch, context length, embedding size

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        query_key_result = (q @ k.transpose(1, 2))/(self.head_size ** 0.5) # (B, C, head_size) x (B, head_size, C) = (batch, C, C)
        query_key_result = query_key_result.masked_fill(torch.tril(torch.ones(C, C, device=device)) == 0, float('-inf'))
        query_key_result = self.softmax(query_key_result)

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

    def forward(self, x):
        # after text and position embeddings and layer norm are done
        x = torch.cat([head(x) for head in self.attn_heads], dim=-1) # (B, C, head_size*num_heads)
        x = self.proj(x) # (B, C, head_size*num_heads) x (head_size*num_heads, E) = (B, C, E)

        return x

class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.l1 = nn.Linear(embedding_size, 4*embedding_size, device=device) # B, C, 4*E
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4*embedding_size, embedding_size, device=device) # B, C, E

    def forward(self, x):

        # x after passing through attention and going through layer norm
        return self.l2(self.relu(self.l1(x)))

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

    def generate(self, max_tokens, token_to_char, max_context_length, start_token=18):
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
model = GPT(vocab_size=vocab_size, max_context_length=MAX_LENGTH, embedding_size=EMBEDDING_SIZE, num_heads=NUM_HEADS, head_size=HEAD_SIZE, num_decoder_layers=NUM_DECODER_BLOCKS)

# model.load_state_dict(torch.load("./full_transformer.pt"))

print("Generate: ")
model.generate(500, token_to_char=token_to_char, max_context_length=MAX_LENGTH, start_token=0)
model.generate(500, token_to_char=token_to_char, max_context_length=MAX_LENGTH, start_token=[tokenizer.get(char) for char in "ROMEO" ])
print("\n")