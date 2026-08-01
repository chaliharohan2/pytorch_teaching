import torch 
import torch.nn as nn

device = torch.device("cuda")
MAX_LENGTH = 10
BATCH_SIZE = 32
EMBEDDING_SIZE = 200
HEAD_SIZE = 400

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embedding_size, device=device)
        self.w_q = nn.Linear(embedding_size, HEAD_SIZE, device=device)
        self.w_k = nn.Linear(embedding_size, HEAD_SIZE, device=device)
        self.relu = nn.ReLU()
        self.w_v = nn.Linear(embedding_size, HEAD_SIZE, device=device)
        self.proj = nn.Linear(HEAD_SIZE, embedding_size, device=device)
        self.softmax = nn.Softmax(dim=-1)
        self.l1 = nn.Linear(embedding_size, 4 * embedding_size, device=device)
        self.l2 = nn.Linear(4 * embedding_size, embedding_size, device=device)
        self.l_out = nn.Linear(embedding_size, vocab_size, device=device)
        self.layer_norm = nn.LayerNorm(embedding_size, device=device)

    def forward(self, data_in):

        # get embedding
        x = self.embedding_table(data_in) 
        B, C, E = x.shape # Batch, length of context, embedding size for each vector

        # calculate query, key, value vectors
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # attention calculation
        query_key_rez = (q @ k.transpose(1, 2))/(HEAD_SIZE**0.5) # batch, length of context, length of context
        tril = torch.tril(torch.ones(C, C, device=device))
        query_key_rez = query_key_rez.masked_fill(tril==0,float('-inf'))
        query_key_rez = self.softmax(query_key_rez)
        
        attn_output = query_key_rez @ v
        attn_proj = self.proj(attn_output) # project attention output to same dimension as embedding vector

        # attention block
        x = self.layer_norm(x + attn_proj) # adding back x for residual connection 

        # feed forward
        ff_output = self.l2(self.relu(self.l1(x)))

        # feedforward block
        x = self.layer_norm(x + ff_output) # adding back x for residual connection 

        # final logits
        logits = self.l_out(x)

        return logits
    
    def generate(self, max_tokens, token_to_char, start_token=18):

        with torch.no_grad():
            sequence = torch.tensor([start_token], device=device).unsqueeze(0)
            for _ in range(max_tokens):
                preds = self(sequence) # 1, context, vocab_size
                preds = preds[:, -1, :] # only take the probs of the last item in the sequence/context (1, vocab_size)
                preds = self.softmax(preds)
                pred_indx = torch.multinomial(preds, num_samples=1)
                pred_char = token_to_char.get(pred_indx[-1, :].item()) # sample from the last dimension of probabilites (vocab_size)
                print(pred_char, end="", flush=True)
                sequence = torch.cat((sequence, pred_indx), dim=1)



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

model = SimpleTransformer(vocab_size=vocab_size, embedding_size=EMBEDDING_SIZE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-2)

for epoch in range(1, 100):

    model.train()
    epoch_loss_vals = []

    for data_in, data_out in dataset:

        optimizer.zero_grad()

        pred = model(data_in)

        B, T, C = pred.shape

        loss = loss_fn(pred.view(B*T, C), data_out.view(B*T))

        epoch_loss_vals.append(loss.item())

        loss.backward()
        optimizer.step()

    
    print("Epoch ", epoch, "\nAvg loss: ", sum(epoch_loss_vals)/float(len(epoch_loss_vals)), "\n")
    print("Generate: ")
    model.generate(100, token_to_char=token_to_char)
    print("\n")