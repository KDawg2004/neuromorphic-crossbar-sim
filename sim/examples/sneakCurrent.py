import json
import time
import numpy as np

from sim.crossbar.builders import build_crossbar
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

R_VALUES = [0.0, 10.0, 50.0, 100.0]
CV_FIXED = 0.10
N_TRIALS = 25
DEVICE_TYPES = ["team", "biolek", "team", "biolek"]
OUT_PATH = "sim/training/sneak_comparison_results.json"


def run_one_solve(cb, solve_fn):
    """Run full-dataset inference using the given per-sample solve function
    (either cb.forward_step-based or the decoupled variant). Returns accuracy.
    NOTE: caller must pass a freshly-programmed crossbar; this function
    reprograms per sample to avoid read-disturb drift, consistent with
    every other sweep in this repo."""
    programmer = CrossbarProgrammer()
    correct = 0
    for i in range(len(X)):
        programmer.map_weights(cb, W)
        cb.apply_row_inputs(X[i])  # confirm this is the correct call to set row_inputs -- see note below
        currents = solve_fn(dt)
        i_plus = currents[0::2]
        i_minus = currents[1::2]
        out = i_plus - i_minus
        if not np.all(np.isfinite(out)):
            continue
        pred = np.argmax(out)
        if pred == y[i]:
            correct += 1
    return correct / len(X)


def run_single_r(r_parasitic, seed_base, max_retries=5):
    coupled_accs = []
    decoupled_accs = []
    total_rejected = 0

    for trial in range(N_TRIALS):
        seed = seed_base + trial
        rejected = 0
        for attempt in range(max_retries):
            try:
                cb_coupled = build_crossbar(
                    in_features, out_features,
                    R_row=r_parasitic, R_col=r_parasitic,
                    device_types=DEVICE_TYPES,
                    variability_cv=CV_FIXED,
                    seed=seed + attempt * 999983,
                )
                cb_decoupled = build_crossbar(
                    in_features, out_features,
                    R_row=r_parasitic, R_col=r_parasitic,
                    device_types=DEVICE_TYPES,
                    variability_cv=CV_FIXED,
                    seed=seed + attempt * 999983,  # SAME seed -> same device draws
                )
                break
            except ValueError:
                rejected += 1
        else:
            total_rejected += rejected
            continue
        total_rejected += rejected

        coupled_accs.append(run_one_solve(cb_coupled, cb_coupled.forward_step))
        decoupled_accs.append(run_one_solve(cb_decoupled, cb_decoupled.solve_node_voltages_decoupled))

    return coupled_accs, decoupled_accs, total_rejected


def load_existing_results():
    try:
        with open(OUT_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_results(results):
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run_sweep():
    results = load_existing_results()
    if results:
        print(f"Resuming: {len(results)} cell(s) already completed, will skip those.")

    sweep_start = time.time()

    for r_parasitic in R_VALUES:
        key = f"r={r_parasitic:.1f}"
        if key in results:
            print(f"{key}: already done, skipping")
            continue

        cell_start = time.time()
        seed_base = int(r_parasitic * 10) * 100000
        coupled, decoupled, rejected = run_single_r(r_parasitic, seed_base)

        elapsed = time.time() - cell_start
        elapsed_total = time.time() - sweep_start
        timestamp = time.strftime("%H:%M:%S")

        if not coupled or not decoupled:
            print(f"{timestamp} {key}: ALL TRIALS REJECTED, skipping (cell took {elapsed/60:.1f} min)")
            results[key] = {"coupled": [], "decoupled": []}
            save_results(results)
            continue

        c_mean, d_mean = np.mean(coupled), np.mean(decoupled)
        delta = d_mean - c_mean
        print(f"{timestamp} {key}: coupled_mean={c_mean:.4f}, decoupled_mean={d_mean:.4f}, "
              f"sneak_path_cost={delta:.4f}, n={len(coupled)}/{N_TRIALS}, "
              f"rejected_draws={rejected}, cell_time={elapsed/60:.1f}min, "
              f"total_elapsed={elapsed_total/60:.1f}min")

        results[key] = {"coupled": coupled, "decoupled": decoupled}
        save_results(results)

    return results


if __name__ == "__main__":
    results = run_sweep()
    print(f"\nDone. Saved raw results to {OUT_PATH}")