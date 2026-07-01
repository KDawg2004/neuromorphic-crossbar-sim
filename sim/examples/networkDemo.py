import numpy as np
from sim.crossbar.crossbar import Crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.network import NeuralNetwork
from devices.TeamMemristor import TEAMMemristor

dt = 1e-4

cb = Crossbar(
    rows=2,
    cols=2
)

for row in range(2):
    for col in range(2):
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

layer = CrossbarLayer(cb)

network = NeuralNetwork()
network.add_layer(layer)

x = np.array([
    1.0,
    0.5
])

y = network.forward(x, dt)

print("Input:")
print(x)

print()

print("Output:")
print(y)