import torch 
import torch.nn as nn

device = torch.device("cuda")

MAX_LENGTH = 50
BATCH_SIZE = 32
EMBEDDING_SIZE = 200
HEAD_SIZE = 20
NUM_HEADS = 10
NUM_DECODER_BLOCKS = 6

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
        self.final_layer_norm = nn.LayerNorm(embedding_size)
        self.linear_output = nn.Linear(embedding_size, vocab_size, device=device)

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

with open("/home/nz-dgx-spark-01/Documents/Nyalazone/pytorch_testing/shakespeare_dataset.txt", mode="r") as f:
    raw_data = f.read()

raw_unique_chars = list(sorted(set(raw_data)))

tokenizer = {char: i for i, char in enumerate(raw_unique_chars)}
token_to_char = {i: char for i, char in enumerate(raw_unique_chars)}
encode = lambda x: [tokenizer.get(c) for c in x]

vocab_size = len(raw_unique_chars)

j = 0
raw_dataset = []

while j < len(raw_data):

    # skip sequences whose length is not equal to max length (usually will be end sequences)
    if len(raw_data[j:j+MAX_LENGTH]) == MAX_LENGTH and len(raw_data[j+1:j+1+MAX_LENGTH]) == MAX_LENGTH:
        new_data = [torch.tensor(encode(raw_data[j:j+MAX_LENGTH]), device=device), torch.tensor(encode(raw_data[j+1:j+1+MAX_LENGTH]), device=device)]
        raw_dataset.append(new_data)

    j += 1

dataset = []

k = 0
while k < len(raw_dataset):
    batch = raw_dataset[k:k+BATCH_SIZE]

    in_tokens = torch.stack([data[0] for data in batch])
    out_tokens = torch.stack([data[1] for data in batch])

    dataset.append([in_tokens, out_tokens])
    
    k += BATCH_SIZE
