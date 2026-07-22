import torch
import torch.nn as nn

device = torch.device("cpu")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.l1(x)
    
model = SimpleModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01)

for name, param in model.named_parameters():
    print(name, param)
    
a = [torch.tensor(i, device=device, dtype=torch.float32) for i in range(101)]
b = [torch.tensor((2*i + 1), device=device, dtype=torch.float32) for i in range(101)]


final_data = list(zip(a, b))

for epoch in range(1, 50):
    
    print("*"*100, f"---- Epoch {epoch} ----", "*"*50)
    model.train()

    loss_vals = []

    for x, y_hat in final_data:

        optimizer.zero_grad()
        y = model(x.unsqueeze(0))
        loss = loss_fn(y, y_hat.unsqueeze(0))
        loss_vals.append(loss.item())

        print("Data in: ", x, " Predicted: ", y.item(), " Data out: ", y_hat)
        loss.backward()
        optimizer.step()

    for name, param in model.named_parameters():
        print(name, param)

    print("Average loss: ", sum(loss_vals)/float(len(loss_vals)), "\n")



