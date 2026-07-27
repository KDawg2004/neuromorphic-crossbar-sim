import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    results_file = root / "sim" / "training" / "variability_sweep_results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    # Sort CV values numerically
    cv_values = sorted(float(k) for k in results.keys())

    means = []
    mins = []
    maxs = []

    for cv in cv_values:
        accs = np.array(results[str(cv)])

        means.append(np.mean(accs))
        mins.append(np.min(accs))
        maxs.append(np.max(accs))

    means = np.array(means)
    mins = np.array(mins)
    maxs = np.array(maxs)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Mean accuracy
    ax.plot(
        cv_values,
        means,
        marker="o",
        linewidth=2,
        label="Mean accuracy"
    )

    # Min-max envelope
    ax.fill_between(
        cv_values,
        mins,
        maxs,
        alpha=0.18,
        label="Min–max range"
    )

    # Chance accuracy
    ax.axhline(
        0.25,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Chance (4-class)"
    )

    ax.set_xlabel("Device-to-device variability (CV)")
    ax.set_ylabel("Inference accuracy")
    ax.set_title(
        "Inference accuracy vs. device variability\n"
        "(mean across 25 trials, shaded band = min–max)"
    )

    # Show the full accuracy range instead of zooming in
    ax.set_ylim(0.20, 1.00)

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    outfile = Path(__file__).with_name("variability_sweep.png")
    plt.savefig(outfile, dpi=300)
    print(f"Saved plot to {outfile}")


if __name__ == "__main__":
    main()