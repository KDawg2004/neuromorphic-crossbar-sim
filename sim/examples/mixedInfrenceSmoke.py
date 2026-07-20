import numpy as np
from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.network import NeuralNetwork
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

# 50/50 split: half the output columns TEAM, half Biolek
#device_types = ["team" if c % 2 == 0 else "biolek" for c in range(out_features)]
device_types = ["team", "team", "team", "biolek"]
print("device_types:", device_types)

cb = build_crossbar(in_features, out_features, R_row=0.0, R_col=0.0, device_types=device_types)

programmer = CrossbarProgrammer()

layer = CrossbarLayer(cb)
network = NeuralNetwork()
network.add_layer(layer)

preds = []
raw_outputs = []
nan_or_inf_count = 0

for i in range(len(X)):
    programmer.map_weights(cb, W)  # reset devices to clean mapped state before each sample
    out = network.forward(X[i], dt)

    if not np.all(np.isfinite(out)):
        nan_or_inf_count += 1
        print(f"sample {i}: NON-FINITE output: {out}")

    raw_outputs.append(out)
    pred = np.argmax(out)
    preds.append(pred)

print("Predictions:", preds)
print("Labels:     ", y.tolist())

correct = sum(p == t for p, t in zip(preds, y.tolist()))
accuracy = correct / len(X)
print(f"Mixed-device crossbar accuracy: {accuracy:.4f} ({correct}/{len(X)})")
print(f"Non-finite output count: {nan_or_inf_count} / {len(X)}")

# sanity: outputs shouldn't all be identical (would suggest devices aren't actually differentiating)
all_outputs = np.array(raw_outputs)
print("Output std across samples per class:", all_outputs.std(axis=0))