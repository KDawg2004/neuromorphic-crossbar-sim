import numpy as np
import torch
import torch.nn as nn
from sim.training.toydataset import X, y

class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 4, bias=False)

    def forward(self, x):
        return self.fc(x)

X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y)

model = ToyNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(2000):
    optimizer.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")

preds = model(X_train).argmax(dim=1)
accuracy = (preds == y_train).float().mean()
print("Final train accuracy:", accuracy.item())

W = model.fc.weight.detach().numpy().T  # transpose to (in, out) for crossbar convention
print("Weight matrix shape:", W.shape)

np.save("sim/training/trained_weights.npy", W)
print("Saved to sim/training/trained_weights.npy")