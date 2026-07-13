import numpy as np
from sim.crossbar.crossbar import Crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.network import NeuralNetwork
from devices.TeamMemristor import TEAMMemristor
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

cb = Crossbar(rows=in_features, cols=out_features * 2)

for row in range(cb.rows):
    for col in range(cb.cols):
        cb.set_device(
            row,
            col,
            TEAMMemristor(
                k_off=1.333,
                k_on=-1.333,
                alpha_off=2,
                alpha_on=2,
                i_off=0.5e-3,
                i_on=-0.5e-3,
                G_on=1/500,
                G_off=1/5000
            )
        )

programmer = CrossbarProgrammer()
programmer.map_weights(cb, W)  # clip to 99th percentile to avoid outlier rows dominating

layer = CrossbarLayer(cb)
network = NeuralNetwork()
network.add_layer(layer)

correct = 0
preds = []

for i in range(len(X)):
    programmer.map_weights(cb, W)  # reset devices to clean mapped state before each sample
    out = network.forward(X[i], dt)
    pred = np.argmax(out)
    preds.append(pred)
    if pred == y[i]:
        correct += 1

accuracy = correct / len(X)
print("Predictions:", preds)
print("Labels:     ", y.tolist())
print(f"Crossbar accuracy: {accuracy:.4f} ({correct}/{len(X)})")
