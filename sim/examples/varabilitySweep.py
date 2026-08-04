import numpy as np
import json
from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.network import NeuralNetwork
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

# --- Sweep configuration ---
CV_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
N_TRIALS = 25
DEVICE_TYPES = ["team", "biolek", "team", "biolek"]  # fixed mixed architecture under test
R_ROW = 0.0
R_COL = 0.0
OUT_PATH = "sim/training/variability_sweep_results.json"


def run_single_trial(cv, seed, max_retries=5):
    """Build one crossbar at the given variability level and seed, run inference,
    return accuracy for this single trial, or None if all retries were rejected."""
    rejected = 0
    for attempt in range(max_retries):
        try:
            cb = build_crossbar(
                in_features, out_features,
                R_row=R_ROW, R_col=R_COL,
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
        programmer.map_weights(cb, W)
        out = network.forward(X[i], dt)

        if not np.all(np.isfinite(out)):
            print(f"  WARNING: non-finite output at cv={cv}, seed={seed}, sample={i}: {out}")
            continue

        pred = np.argmax(out)
        if pred == y[i]:
            correct += 1

    return correct / len(X), rejected


def load_existing_results():
    try:
        with open(OUT_PATH, "r") as f:
            raw = json.load(f)
        # keys were saved as strings, convert back to float for lookup consistency
        return {float(k): v for k, v in raw.items()}
    except FileNotFoundError:
        return {}


def save_results(results):
    with open(OUT_PATH, "w") as f:
        json.dump({str(cv): accs for cv, accs in results.items()}, f, indent=2)


def run_sweep():
    results = load_existing_results()
    if results:
        print(f"Resuming: {len(results)} cell(s) already completed, will skip those.")

    for cv in CV_VALUES:
        if cv in results:
            print(f"cv={cv:.2f}: already done, skipping")
            continue

        accuracies = []
        total_rejected = 0
        for trial in range(N_TRIALS):
            # deterministic seed, stable across processes -- do NOT use
            # Python's hash() on a tuple here, it's per-process salted
            # and not reproducible run-to-run (see project notes)
            seed = int(cv * 1000) * 1000 + trial
            acc, rejected = run_single_trial(cv, seed)
            total_rejected += rejected
            if acc is not None:
                accuracies.append(acc)

        if not accuracies:
            print(f"cv={cv:.2f}: ALL TRIALS REJECTED, skipping")
            results[cv] = []
            save_results(results)
            continue

        results[cv] = accuracies
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(f"cv={cv:.2f}: mean={mean:.4f}, std={std:.4f}, "
              f"min={min(accuracies):.4f}, max={max(accuracies):.4f}, "
              f"n={len(accuracies)}/{N_TRIALS}, rejected_draws={total_rejected}")

        save_results(results)  # save after every cv, not just at the end

    return results


if __name__ == "__main__":
    results = run_sweep()
    print(f"\nDone. Saved raw results to {OUT_PATH}")