import numpy as np
from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.network import NeuralNetwork
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

device_types = ["team", "team", "team", "biolek"]  # 75/25

cb = build_crossbar(in_features, out_features, R_row=0.0, R_col=0.0, device_types=device_types)
programmer = CrossbarProgrammer()
layer = CrossbarLayer(cb)
network = NeuralNetwork()
network.add_layer(layer)

n_classes = out_features
confusion = np.zeros((n_classes, n_classes), dtype=int)  # rows=true, cols=predicted

for i in range(len(X)):
    programmer.map_weights(cb, W)
    out = network.forward(X[i], dt)
    pred = int(np.argmax(out))
    true = int(y[i])
    confusion[true, pred] += 1

print("Confusion matrix (rows=true label, cols=predicted label):")
print("        " + "".join(f"pred{c:>6}" for c in range(n_classes)))
for true_c in range(n_classes):
    row = confusion[true_c]
    total = row.sum()
    acc = row[true_c] / total if total > 0 else 0.0
    print(f"true {true_c}: " + "".join(f"{v:>10}" for v in row) + f"   (recall={acc:.3f}, n={total})")

print()
print("Per-class recall:")
for c in range(n_classes):
    total = confusion[c].sum()
    recall = confusion[c, c] / total if total > 0 else 0.0
    print(f"  class {c}: {recall:.3f}")

print()
print("How often each class was over-predicted (col sum minus diagonal, i.e. false positives):")
for c in range(n_classes):
    col = confusion[:, c]
    false_pos = col.sum() - col[c]
    print(f"  class {c}: {false_pos} false positives out of {800 - col.sum() + false_pos + col[c] - col[c]}")