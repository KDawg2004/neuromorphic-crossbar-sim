import numpy as np
from sim.crossbar.builders import build_crossbar
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

device_types = ["team", "biolek", "team", "biolek"]

cb = build_crossbar(in_features, out_features, R_row=0.0, R_col=0.0,
                     device_types=device_types, variability_cv=0.0)
programmer = CrossbarProgrammer()
programmer.map_weights(cb, W)

# --- snapshot state before any forward_step ---
def snapshot():
    states = []
    for r in range(cb.rows):
        row = []
        for c in range(cb.cols):
            d = cb.get_device(r, c)
            val = d.w_init if hasattr(d, "w_init") else d.x_init
            row.append(val)
        states.append(row)
    return np.array(states)

before = snapshot()

# --- one forward_step, single sample, no reprogramming ---
cb.apply_row_inputs(X[0])
_ = cb.forward_step(dt)

after = snapshot()

delta = after - before
abs_delta = np.abs(delta)

print("=== Read disturb check: single forward_step, single sample ===")
print(f"max |delta|:  {abs_delta.max():.3e}")
print(f"mean |delta|: {abs_delta.mean():.3e}")
print(f"nonzero deltas: {np.count_nonzero(abs_delta > 1e-15)} / {abs_delta.size}")

# is this floating point noise or real drift? compare against machine epsilon scale
eps_scale = np.finfo(np.float64).eps * 10  # generous noise floor
n_above_noise_floor = np.count_nonzero(abs_delta > eps_scale)
print(f"deltas above float64 noise floor ({eps_scale:.1e}): {n_above_noise_floor} / {abs_delta.size}")

# --- input voltage magnitude vs device thresholds ---
print("\n=== Input voltage vs switching thresholds ===")
print(f"X[0] range: min={X[0].min():.4f}, max={X[0].max():.4f}, mean={np.abs(X[0]).mean():.4f}")

# grab actual threshold params from one device of each type
team_dev = next(cb.get_device(r, c) for r in range(cb.rows) for c in range(cb.cols)
                 if type(cb.get_device(r, c)).__name__ == "TEAMMemristor")
biolek_dev = next(cb.get_device(r, c) for r in range(cb.rows) for c in range(cb.cols)
                   if type(cb.get_device(r, c)).__name__ == "BiolekMemcapacitor")

print(f"TEAM i_off={team_dev.i_off:.3e}, i_on={team_dev.i_on:.3e}")
print(f"Biolek k={biolek_dev.k:.3e}, p={biolek_dev.p}")

# --- accumulated drift across a full 800-sample pass, no reprogramming between samples ---
print("\n=== Accumulated drift across full dataset (no reprogramming between samples) ===")
start = snapshot()
for i in range(len(X)):
    cb.apply_row_inputs(X[i])
    _ = cb.forward_step(dt)
end = snapshot()

full_delta = np.abs(end - start)
print(f"max |delta| after {len(X)} samples: {full_delta.max():.3e}")
print(f"mean |delta| after {len(X)} samples: {full_delta.mean():.3e}")
print(f"fraction of devices with |delta| > 1e-6: "
      f"{np.count_nonzero(full_delta > 1e-6) / full_delta.size:.4f}")