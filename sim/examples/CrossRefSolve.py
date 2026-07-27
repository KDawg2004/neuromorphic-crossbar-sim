import numpy as np
from sim.crossbar.builders import build_crossbar
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X

dt = 1e-4

W1 = np.load("sim/training/trained_weights_l1.npy")
in1, out1 = W1.shape

cb = build_crossbar(
    in1, out1, R_row=10.0, R_col=10.0,  # nonzero R so the wire-chain terms are actually exercised
    device_types=["team", "biolek"] * (out1 // 2),
    variability_cv=0.0,
    seed=1,
)

programmer = CrossbarProgrammer()
programmer.map_weights(cb, W1)
cb.apply_row_inputs(X[0])

V_dense = cb.solve_node_voltages_REF(dt)
V_sparse = cb.solve_node_voltages(dt)

diff = np.abs(V_dense - V_sparse)
print("max abs diff:", diff.max())
print("mean abs diff:", diff.mean())
print("dense sample:", V_dense[:5])
print("sparse sample:", V_sparse[:5])