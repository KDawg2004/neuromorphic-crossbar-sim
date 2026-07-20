import json
import time
import numpy as np

from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.network import NeuralNetwork
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

# --- Sweep configuration ---
CV_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]
R_VALUES = [0.0, 10.0, 50.0, 100.0]  # ohms, confirmed range
N_TRIALS = 25
DEVICE_TYPES = ["team", "biolek", "team", "biolek"]
OUT_PATH = "sim/training/parasitic_variability_grid_results.json"


def get_state_hash(cb, rows, cols):
    """Diagnostic: hash all device internal states to detect unwanted mutation
    during inference. cols = out_features * 2 (differential pairs)."""
    states = []
    for r in range(rows):
        for c in range(cols):
            device = cb.get_device(r, c)
            state = device.w_init if hasattr(device, "w_init") else device.x
            states.append(state)
    return hash(tuple(np.round(states, 12)))


def run_single_trial(cv, r_parasitic, seed, max_retries=5):
    rejected = 0
    for attempt in range(max_retries):
        try:
            cb = build_crossbar(
                in_features, out_features,
                R_row=r_parasitic, R_col=r_parasitic,
                device_types=DEVICE_TYPES,
                variability_cv=cv,
                seed=seed + attempt * 999983,
            )
            break
        except ValueError:
            rejected += 1
    else:
        return None, rejected

    programmer = CrossbarProgrammer()
    layer = CrossbarLayer(cb)
    network = NeuralNetwork()
    network.add_layer(layer)

    correct = 0
    for i in range(len(X)):
        # map_weights must run per-sample, not once per crossbar build.
        # forward_step applies read voltages through the same TEAM/Biolek
        # dynamics as programming (no read/write regime separation in the
        # device models), so inference causes real read disturb.
        # Re-programming before every sample resets state to the clean
        # baseline and is what makes reported accuracy trustworthy.
        # Confirmed via device-state hash diagnostic, session 8. Do not
        # move this outside the loop for a speedup, it silently corrupts
        # every result past the first sample.
        programmer.map_weights(cb, W)
        out = network.forward(X[i], dt)
        if not np.all(np.isfinite(out)):
            continue
        pred = np.argmax(out)
        if pred == y[i]:
            correct += 1

    return correct / len(X), rejected


def load_existing_results():
    try:
        with open(OUT_PATH, "r") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items()}
    except FileNotFoundError:
        return {}


def save_results(results):
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_grid():
    results = load_existing_results()
    if results:
        print(f"Resuming: {len(results)} cell(s) already completed, will skip those.")

    grid_start = time.time()
    total_cells = len(R_VALUES) * len(CV_VALUES)
    cell_num = 0

    for r_parasitic in R_VALUES:
        for cv in CV_VALUES:
            cell_num += 1
            key = f"cv={cv:.2f}_r={r_parasitic:.1f}"

            if key in results:
                print(f"[{cell_num}/{total_cells}] {key}: already done, skipping")
                continue

            cell_start = time.time()
            accuracies = []
            total_rejected = 0

            for trial in range(N_TRIALS):
                seed = int(cv * 1000) * 1000 + int(r_parasitic * 10) * 100000 + trial
                acc, rejected = run_single_trial(cv, r_parasitic, seed)
                total_rejected += rejected
                if acc is not None:
                    accuracies.append(acc)

            elapsed = time.time() - cell_start
            elapsed_total = time.time() - grid_start
            timestamp = time.strftime("%H:%M:%S")

            if not accuracies:
                print(f"[{cell_num}/{total_cells}] {timestamp} {key}: ALL TRIALS REJECTED, skipping "
                      f"(cell took {elapsed/60:.1f} min)")
                results[key] = []
                save_results(results)
                continue

            mean = np.mean(accuracies)
            std = np.std(accuracies)
            print(f"[{cell_num}/{total_cells}] {timestamp} {key}: mean={mean:.4f}, std={std:.4f}, "
                  f"min={min(accuracies):.4f}, max={max(accuracies):.4f}, "
                  f"n={len(accuracies)}/{N_TRIALS}, rejected_draws={total_rejected}, "
                  f"cell_time={elapsed/60:.1f}min, total_elapsed={elapsed_total/60:.1f}min")

            results[key] = accuracies
            save_results(results)  # write after every cell, not just at the end

    return results


if __name__ == "__main__":
    results = run_grid()
    print(f"\nDone. Saved raw results to {OUT_PATH}")