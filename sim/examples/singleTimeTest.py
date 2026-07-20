import time
from sim.crossbar.builders import build_crossbar

cb = build_crossbar(64, 4, R_row=10.0, R_col=10.0, device_types=["team","biolek","team","biolek"], variability_cv=0.0, seed=1)
from sim.nn.layer import CrossbarLayer
from sim.nn.mapper import CrossbarProgrammer
import numpy as np

W = np.load("sim/training/trained_weights.npy")
programmer = CrossbarProgrammer()
layer = CrossbarLayer(cb)
programmer.map_weights(cb, W)

from sim.training.toydataset import X
t0 = time.time()
out = layer.forward(X[0], 1e-4)
print("one sample:", time.time() - t0, "seconds")