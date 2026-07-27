import json
import time
import numpy as np

from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W1 = np.load("sim/training/trained_weights_l1.npy")  # (64, 16)
W2 = np.load("sim/training/trained_weights_l2.npy")  # (16, 4)
in1, out1 = W1.shape
in2, out2 = W2.shape

CV_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]
N_TRIALS = 25
DEVICE_TYPES_L1 = ["team", "biolek"] * (out1 // 2)  # confirm out1 is even; H=16 -> 8 pairs
DEVICE_TYPES_L2 = ["team", "biolek", "team", "biolek"]  # out2=4, same as before
R_FIXED = 0.0
TARGET_LAYERS = ["layer1", "layer2"]
OUT_PATH = "sim/training/layer_sensitivity_results.json"


def relu(x):
    return np.maximum(x, 0.0)


def run_single_trial(cv, seed, target_layer, max_retries=5):
    cv_l1 = cv if target_layer == "layer1" else 0.0
    cv_l2 = cv if target_layer == "layer2" else 0.0

    rejected = 0
    for attempt in range(max_retries):
        try:
            cb1 = build_crossbar(
                in1, out1, R_row=R_FIXED, R_col=R_FIXED,
                device_types=DEVICE_TYPES_L1, variability_cv=cv_l1,
                seed=seed + attempt * 999983,
            )
            cb2 = build_crossbar(
                in2, out2, R_row=R_FIXED, R_col=R_FIXED,
                device_types=DEVICE_TYPES_L2, variability_cv=cv_l2,
                seed=seed + attempt * 999983 + 1,  # distinct seed, still deterministic
            )
            break
        except ValueError:
            rejected += 1
    else:
        return None, rejected

    programmer = CrossbarProgrammer()
    layer1 = CrossbarLayer(cb1)
    layer2 = CrossbarLayer(cb2)

    correct = 0
    for i in range(len(X)):
        programmer.map_weights(cb1, W1)
        programmer.map_weights(cb2, W2)

        x1 = layer1.forward(X[i], dt)
        x1_relu = relu(x1)
        out = layer2.forward(x1_relu, dt)

        if not np.all(np.isfinite(out)):
            continue
        pred = np.argmax(out)
        if pred == y[i]:
            correct += 1

    return correct / len(X), rejected


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

    for target_layer in TARGET_LAYERS:
        for cv in CV_VALUES:
            key = f"{target_layer}_cv={cv:.2f}"
            if key in results:
                print(f"{key}: already done, skipping")
                continue

            cell_start = time.time()
            accuracies = []
            total_rejected = 0

            for trial in range(N_TRIALS):
                seed = int(cv * 1000) * 1000 + trial
                acc, rejected = run_single_trial(cv, seed, target_layer)
                total_rejected += rejected
                if acc is not None:
                    accuracies.append(acc)

            elapsed = time.time() - cell_start
            elapsed_total = time.time() - sweep_start
            timestamp = time.strftime("%H:%M:%S")

            if not accuracies:
                print(f"{timestamp} {key}: ALL TRIALS REJECTED, skipping (cell took {elapsed/60:.1f} min)")
                results[key] = []
                save_results(results)
                continue

            mean = np.mean(accuracies)
            std = np.std(accuracies)
            print(f"{timestamp} {key}: mean={mean:.4f}, std={std:.4f}, "
                  f"min={min(accuracies):.4f}, max={max(accuracies):.4f}, "
                  f"n={len(accuracies)}/{N_TRIALS}, rejected_draws={total_rejected}, "
                  f"cell_time={elapsed/60:.1f}min, total_elapsed={elapsed_total/60:.1f}min")

            results[key] = accuracies
            save_results(results)

    return results


if __name__ == "__main__":
    results = run_sweep()
    print(f"\nDone. Saved raw results to {OUT_PATH}")