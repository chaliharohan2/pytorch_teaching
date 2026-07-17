import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cpu")
max_length = 10
batch_size = 32

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) # batch, max_length, vocab_size

        return x
    
    def generate(self, max_tokens, token_to_char, start_token=18, temperature=2.0):

        with torch.no_grad():
            sequence = torch.tensor([start_token], device=device).unsqueeze(0)
            softmax_layer = nn.Softmax(dim=-1)

            for _ in range(max_tokens):
                preds = softmax_layer(self(sequence)/temperature)
                preds = preds.squeeze(0)
                pred_token_idx = torch.multinomial(preds, num_samples=1)
                pred_char = token_to_char.get(pred_token_idx.squeeze(0).item())
                print(pred_char, end="", flush=True)
                sequence = pred_token_idx

    

with open("/Users/rohan/Nyalazone/pytorch_testing/shakespeare_dataset.txt", mode="r") as f:
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
    if len(raw_data[j:j+max_length]) == max_length and len(raw_data[j+1:j+1+max_length]) == max_length:
        new_data = [torch.tensor(encode(raw_data[j:j+max_length]), device=device), torch.tensor(encode(raw_data[j+1:j+1+max_length]), device=device)]
        raw_dataset.append(new_data)

    j += 1

dataset = []

k = 0
while k < len(raw_dataset):
    batch = raw_dataset[k:k+batch_size]

    in_tokens = torch.stack([data[0] for data in batch])
    out_tokens = torch.stack([data[1] for data in batch])

    dataset.append([in_tokens, out_tokens])
    
    k += batch_size

model = BigramModel(vocab_size=vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-2)

for epoch in range(1, 1000):

    model.train()
    epoch_loss_vals = []

    for in_tokens, out_tokens in dataset:


        optimizer.zero_grad()

        preds = model(in_tokens)
        B, T, C = preds.shape

        loss = loss_fn(preds.view(B*T, C), out_tokens.view(B*T))
        
        epoch_loss_vals.append(loss.item())

        loss.backward()
        optimizer.step()

    avg_loss = float(sum(epoch_loss_vals)) / len(epoch_loss_vals)

    print("\nAvg loss for epoch ", epoch, " = ", avg_loss)

    print("Generating sequence: \n")
    model.eval()
    model.generate(max_tokens=100, token_to_char=token_to_char)


