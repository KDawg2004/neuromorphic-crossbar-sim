import numpy as np
import torch
import torch.nn as nn

from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X

dt = 1e-4

W1 = np.load("sim/training/trained_weights_l1.npy")  # (64, 16)
in1, out1 = W1.shape

# --- crossbar side ---
cb1 = build_crossbar(
    in1, out1, R_row=0.0, R_col=0.0,
    device_types=["team", "biolek"] * (out1 // 2),
    variability_cv=0.0,
    seed=1,
)
programmer = CrossbarProgrammer()
layer1 = CrossbarLayer(cb1)
programmer.map_weights(cb1, W1)

crossbar_outputs = []
for i in range(5):
    out = layer1.forward(X[i], dt)
    crossbar_outputs.append(out)
crossbar_outputs = np.array(crossbar_outputs)

# --- pytorch side, using the SAME W1 the crossbar was programmed with ---
fc1 = nn.Linear(64, 16, bias=False)
with torch.no_grad():
    fc1.weight.copy_(torch.from_numpy(W1.T).float())  # W1 is (in,out), nn.Linear wants (out,in)

X_torch = torch.from_numpy(X[:5]).float()
torch_outputs = fc1(X_torch).detach().numpy()

print("Crossbar layer1 output range:", crossbar_outputs.min(), "to", crossbar_outputs.max())
print("PyTorch fc1 output range:    ", torch_outputs.min(), "to", torch_outputs.max())
print()
print("Crossbar sample[0]:", crossbar_outputs[0])
print("PyTorch  sample[0]:", torch_outputs[0])