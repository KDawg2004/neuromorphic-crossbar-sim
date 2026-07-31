import numpy as np
import torch
import torch.nn as nn

from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X

dt = 1e-4

W1 = np.load("sim/training/trained_weights_l1.npy")
W2 = np.load("sim/training/trained_weights_l2.npy")
in1, out1 = W1.shape
in2, out2 = W2.shape

DEVICE_TYPES_L1 = ["team", "biolek"] * (out1 // 2)  # confirm out1 is even; H=16 -> 8 pairs
def calibrate_activation_scales(W1, in1, out1, device_types_l1, dt, n_samples=20):
    """Calibrate a SEPARATE PGA gain per output-column device type.

    TEAM (roughly linear conductance) and Biolek (steep 1/DM(x) response)
    produce systematically different raw current magnitudes for
    functionally equivalent programmed weights -- confirmed directly by
    comparing crossbar vs PyTorch layer-1 output per neuron: TEAM-driven
    neurons matched PyTorch closely under a single global scale, Biolek-
    driven neurons were attenuated by roughly 3x relative to what a single
    scalar could correct (see project notes). A single global PGA gain is
    therefore a poor compromise across both populations. This computes one
    gain per device type instead, applied per-column by column device type
    at forward-pass time.

    Still a per-device-type CONSTANT calibrated once at nominal (cv=0, R=0)
    conditions -- same fixed-gain-hardware justification as before, just
    two gains instead of one. Does not adapt to variability/endurance.
    """

    cb_cal = build_crossbar(
        in1, out1, R_row=0.0, R_col=0.0,
        device_types=device_types_l1, variability_cv=0.0, seed=0,
    )
    programmer = CrossbarProgrammer()
    layer_cal = CrossbarLayer(cb_cal)
    programmer.map_weights(cb_cal, W1)

    fc1 = nn.Linear(in1, out1, bias=False)
    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(W1.T).float())

    crossbar_vals, torch_vals = [], []
    for i in range(n_samples):
        cb_cal.reset_states()
        crossbar_vals.append(layer_cal.forward(X[i], dt))
        torch_vals.append(fc1(torch.from_numpy(X[i]).float()).detach().numpy())

    crossbar_vals = np.abs(np.array(crossbar_vals))  # (n_samples, out1)
    torch_vals = np.abs(np.array(torch_vals))

    scales = np.zeros(out1)
    for col in range(out1):
        mask = crossbar_vals[:, col] > 1e-9
        if not np.any(mask):
            scales[col] = 1.0  # degenerate fallback, shouldn't normally hit this
            continue
        scales[col] = np.median(torch_vals[mask, col] / crossbar_vals[mask, col])

    team_cols = [c for c in range(out1) if device_types_l1[c] == "team"]
    biolek_cols = [c for c in range(out1) if device_types_l1[c] == "biolek"]
    print(f"TEAM columns {team_cols}: scale = {scales[team_cols].mean():.2f}")
    print(f"Biolek columns {biolek_cols}: scale = {scales[biolek_cols].mean():.2f}")

    return scales  # per-column array, length out1

ACTIVATION_SCALES = calibrate_activation_scales(W1, in1, out1, DEVICE_TYPES_L1, dt)  # paste your calibrated values here
SAMPLE_IDX = 4

cb1 = build_crossbar(in1, out1, R_row=0.0, R_col=0.0,
                      device_types=["team", "biolek"] * (out1 // 2),
                      variability_cv=0.0, seed=1)
print("Rows:", len(cb1.devices))
print("Cols:", len(cb1.devices[0]))
for r in range(2):
    for c in range(8):
        print(r, c, type(cb1.devices[r][c]).__name__)
cb2 = build_crossbar(in2, out2, R_row=0.0, R_col=0.0,
                      device_types=["team", "biolek", "team", "biolek"],
                      variability_cv=0.0, seed=2)

programmer = CrossbarProgrammer()
layer1 = CrossbarLayer(cb1)
layer2 = CrossbarLayer(cb2)
programmer.map_weights(cb1, W1)
row = 0
col = 11   # Biolek neuron from your table

plus = cb1.get_device(row, col * 2)
minus = cb1.get_device(row, col * 2 + 1)

print("Weight =", W1[row, col])
print("Plus x =", plus.x)
print("Minus x =", minus.x)
print("Plus DM =", plus.DM(plus.x))
print("Minus DM =", minus.DM(minus.x))

print("Plus G =", plus.current_conductance(dt))
print("Minus G =", minus.current_conductance(dt))
programmer.map_weights(cb2, W2)

print(type(cb1.devices))
print(type(cb1.devices[0]))
print(type(cb1.devices[0][0]))
print(type(cb1.devices[0][1]))
print(type(cb1.devices[1][0]))
print(type(cb1.devices[1][1]))

def relu(x):
    return np.maximum(x, 0.0)

fc1 = nn.Linear(in1, out1, bias=False)
fc2 = nn.Linear(in2, out2, bias=False)
with torch.no_grad():
    fc1.weight.copy_(torch.from_numpy(W1.T).float())
    fc2.weight.copy_(torch.from_numpy(W2.T).float())

interesting = []
printed = 0
for sample_idx in [2]:
    cb1.reset_states()
    cb2.reset_states()
    x_np = X[sample_idx]
    x_torch = torch.from_numpy(x_np).float()

    # ----- Layer 1 -----
    l1_pt = fc1(x_torch).detach().numpy()
    l1_cb = layer1.forward(x_np, dt) * ACTIVATION_SCALES

    relu_pt = np.maximum(l1_pt, 0)
    relu_cb = np.maximum(l1_cb, 0)

    # ReLU mask disagreement
    pt_sign = l1_pt > 0
    cb_sign = l1_cb > 0

    mask_diff = np.sum(pt_sign != cb_sign)

    # Relative hidden error
    hidden_err = np.mean(np.abs(relu_pt - relu_cb))

    # ----- Layer 2 -----
    out_pt = fc2(torch.from_numpy(relu_pt).float()).detach().numpy()
    out_cb = layer2.forward(relu_cb, dt)

    pt_pred = np.argmax(out_pt)
    cb_pred = np.argmax(out_cb)

    pred_diff = (pt_pred != cb_pred)
    if pred_diff and mask_diff > 0 and printed < 10:
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx}")
        print(f"Prediction: PyTorch={pt_pred}  Crossbar={cb_pred}")
        print(f"{'Neuron':>6} {'Dev':>6} {'PyTorch':>12} {'Crossbar':>12} {'PT':>4} {'CB':>4}")
        print("-" * 52)

        for j in range(out1):
            dev = DEVICE_TYPES_L1[j]

            pt_sign = "+" if l1_pt[j] > 0 else "-"
            cb_sign = "+" if l1_cb[j] > 0 else "-"

            marker = " <--" if pt_sign != cb_sign else ""

            print(
                f"{j:6d} "
                f"{dev:>6} "
                f"{l1_pt[j]:12.5f} "
                f"{l1_cb[j]:12.5f} "
                f"{pt_sign:>4} "
                f"{cb_sign:>4}"
                f"{marker}"
            )

        print()
        printed += 1

    interesting.append({
        "sample": sample_idx,
        "mask_diff": mask_diff,
        "hidden_err": hidden_err,
        "pred_diff": pred_diff,
        "pt_pred": pt_pred,
        "cb_pred": cb_pred,
        "out_pt": out_pt,
        "out_cb": out_cb,
        "relu_pt": relu_pt,
        "relu_cb": relu_cb,
    })
'''
interesting = [
    s for s in interesting
    if s["pred_diff"] or s["mask_diff"] > 0
]
# Sort: prediction mismatches first, then most ReLU disagreements,
# then largest hidden activation error.
interesting.sort(
    key=lambda s: (
        s["pred_diff"],
        s["mask_diff"],
        s["hidden_err"],
    ),
    reverse=True,
)

print(f"\nFound {sum(s['pred_diff'] for s in interesting)} prediction mismatches.\n")

for s in interesting[:20]:
    print("=" * 80)
    print(f"Sample {s['sample']}")
    print(f"Prediction: PyTorch={s['pt_pred']}  Crossbar={s['cb_pred']}")
    print(f"ReLU mask differences: {s['mask_diff']}")
    print(f"Mean hidden error: {s['hidden_err']:.4f}")

    if s["mask_diff"] > 0:
        diff = np.where((s["relu_pt"] > 0) != (s["relu_cb"] > 0))[0]
        print("Different ReLU neurons:", diff)

    print("PyTorch logits :", np.round(s["out_pt"], 3))
    print("Crossbar logits:", np.round(s["out_cb"], 6))
    print()
    '''