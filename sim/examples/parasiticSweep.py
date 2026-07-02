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

def build_crossbar(R_row, R_col):
    cb = Crossbar(rows=in_features, cols=out_features * 2, R_row=R_row, R_col=R_col)
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
    return cb

def evaluate(R_row, R_col):
    cb = build_crossbar(R_row, R_col)
    programmer = CrossbarProgrammer()
    layer = CrossbarLayer(cb)
    network = NeuralNetwork()
    network.add_layer(layer)

    correct = 0
    for i in range(len(X)):
        programmer.map_weights(cb, W)
        out = network.forward(X[i], dt)
        pred = np.argmax(out)
        if pred == y[i]:
            correct += 1
    return correct / len(X)

resistances = [0.0, 1.0, 10.0, 100.0, 1000.0]

for R in resistances:
    acc = evaluate(R, R)
    print(f"R_row=R_col={R} ohm -> accuracy {acc:.4f}")