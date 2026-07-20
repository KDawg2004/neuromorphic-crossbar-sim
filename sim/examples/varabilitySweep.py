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
CV_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]
N_TRIALS = 25
DEVICE_TYPES = ["team", "biolek", "team", "team"]  # fixed mixed architecture under test
R_ROW = 0.0
R_COL = 0.0


def run_single_trial(cv, seed):
    """Build one crossbar at the given variability level and seed, run inference,
    return accuracy for this single trial."""
    cb = build_crossbar(
        in_features, out_features,
        R_row=R_ROW, R_col=R_COL,
        device_types=DEVICE_TYPES,
        variability_cv=cv,
        seed=seed,
    )

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

    return correct / len(X)


def run_sweep():
    results = {}  # cv -> list of accuracies, one per trial

    for cv in CV_VALUES:
        accuracies = []
        for trial in range(N_TRIALS):
            # seed derived from cv and trial index so the whole sweep is reproducible
            # from CV_VALUES/N_TRIALS alone, no external seed state needed
            seed = hash((cv, trial)) % (2**32)
            acc = run_single_trial(cv, seed)
            accuracies.append(acc)

        results[cv] = accuracies
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(f"cv={cv:.2f}: mean={mean:.4f}, std={std:.4f}, "
              f"min={min(accuracies):.4f}, max={max(accuracies):.4f}")

    return results


if __name__ == "__main__":
    results = run_sweep()

    # save raw results for later plotting / re-analysis, not just the printed summary
    out_path = "sim/training/variability_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({str(cv): accs for cv, accs in results.items()}, f, indent=2)
    print(f"\nSaved raw results to {out_path}")