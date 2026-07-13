import numpy as np
from sim.crossbar.builders import build_crossbar
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

device_types = ["team", "biolek", "team", "biolek"]

cb = build_crossbar(in_features, out_features, R_row=0.0, R_col=0.0, device_types=device_types)
programmer = CrossbarProgrammer()

N_SAMPLES = 5

w_max = np.abs(W).max()
print(f"w_max (global, used for normalization): {w_max:.6f}\n")

# --- Static check: what does programming actually set, independent of any sample ---
programmer.map_weights(cb, W)  # clip to 99th percentile to avoid outlier rows dominating

print("=== Programmed state after map_weights (static, from trained weights) ===")
for col_pair, kind in [(1, "biolek col 1"), (3, "biolek col 3")]:
    plus = cb.get_device(0, col_pair * 2)      # row 0 as representative
    minus = cb.get_device(0, col_pair * 2 + 1)
    print(f"{kind}: W[:,{col_pair}] mean={W[:,col_pair].mean():.4f}, "
          f"max|W[:,{col_pair}]|={np.abs(W[:,col_pair]).max():.4f}")
    print(f"  plus  device: x_init={plus.x_init:.6f}, Cmin={plus.Cmin:.3e}, Cmax={plus.Cmax:.3e}")
    print(f"  minus device: x_init={minus.x_init:.6f}, Cmin={minus.Cmin:.3e}, Cmax={minus.Cmax:.3e}")
    print()

# --- Per-row check for column 3 specifically: is one row's weight an outlier ---
print("=== Column 3 per-row weight and programmed state (all 64 rows) ===")
col = 3
plus_states = []
minus_states = []
for row in range(in_features):
    w_norm = W[row, col] / w_max
    mag = abs(w_norm)
    expected_plus = mag if w_norm >= 0 else 0.0
    expected_minus = 0.0 if w_norm >= 0 else mag

    plus = cb.get_device(row, col * 2)
    minus = cb.get_device(row, col * 2 + 1)

    plus_states.append(1.0 - plus.x_init)   # recovered programmed state
    minus_states.append(1.0 - minus.x_init)

    #if row < 10 or abs(W[row, col]) == np.abs(W[:, col]).max():
    print(f"row {row}: W={W[row,col]:+.4f}, expected_plus={expected_plus:.4f}, "
            f"expected_minus={expected_minus:.4f}, "
            f"actual_plus_state={1.0-plus.x_init:.4f}, actual_minus_state={1.0-minus.x_init:.4f}")

plus_states = np.array(plus_states)
minus_states = np.array(minus_states)
print(f"\ncol 3 plus  states: min={plus_states.min():.4f}, max={plus_states.max():.4f}, mean={plus_states.mean():.4f}")
print(f"col 3 minus states: min={minus_states.min():.4f}, max={minus_states.max():.4f}, mean={minus_states.mean():.4f}")

# --- Compare Cmin/Cmax actually assigned across all col-3 devices (variability_cv=0 here so should be identical) ---
print("\n=== Col 3 device Cmin/Cmax consistency (should be identical, variability_cv=0) ===")
cmins_plus = [cb.get_device(r, col*2).Cmin for r in range(in_features)]
cmaxs_plus = [cb.get_device(r, col*2).Cmax for r in range(in_features)]
cmins_minus = [cb.get_device(r, col*2+1).Cmin for r in range(in_features)]
cmaxs_minus = [cb.get_device(r, col*2+1).Cmax for r in range(in_features)]
print(f"plus  Cmin unique values: {set(cmins_plus)}")
print(f"plus  Cmax unique values: {set(cmaxs_plus)}")
print(f"minus Cmin unique values: {set(cmins_minus)}")
print(f"minus Cmax unique values: {set(cmaxs_minus)}")
print("max|W[:,c]| per column:")
for c in range(W.shape[1]):
    print(f"col {c} ({device_types[c]}): max|W| = {np.abs(W[:, c]).max():.6f}")