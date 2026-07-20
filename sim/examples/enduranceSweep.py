import json
import time
import numpy as np

from devices.endurance import degraded_biolek_defaults
from devices.FracMemCap import BIOLEK_DEFAULTS
from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.network import NeuralNetwork
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X, y

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")
in_features, out_features = W.shape

N_CYCLES = [0, 1e3, 1e4, 1e5, 3e5, 6e5, 1e6, 3e6, 1e7, 1e8, 1e10]
CV_FIXED = 0.10
R_FIXED = 0.0
N_TRIALS = 25
DEVICE_TYPES = ["team", "biolek", "team", "biolek"]
OUT_PATH = "sim/training/endurance_sweep_results.json"


def run_single_trial(N_cyc, seed, max_retries=5):
    rejected = 0

    cmin_deg, cmax_deg, cinit_deg = degraded_biolek_defaults(
        N_cyc,
        BIOLEK_DEFAULTS["Cmin"],
        BIOLEK_DEFAULTS["Cmax"],
        BIOLEK_DEFAULTS["Cinit"],
    )
    biolek_params = {"Cmin": cmin_deg, "Cmax": cmax_deg, "Cinit": cinit_deg}

    for attempt in range(max_retries):
        try:
            cb = build_crossbar(
                in_features, out_features,
                R_row=R_FIXED, R_col=R_FIXED,
                device_types=DEVICE_TYPES,
                variability_cv=CV_FIXED,
                seed=seed + attempt * 999983,
                biolek_params=biolek_params,
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

    for N_cyc in N_CYCLES:
        key = f"N={N_cyc:.0e}"

        if key in results:
            print(f"{key}: already done, skipping")
            continue

        cell_start = time.time()
        accuracies = []
        total_rejected = 0

        for trial in range(N_TRIALS):
            seed = int(np.log10(max(N_cyc, 1)) * 1000) * 1000 + trial
            acc, rejected = run_single_trial(N_cyc, seed)
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