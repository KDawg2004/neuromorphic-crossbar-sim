import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = "sim/training/sneak_comparison_results.json"
OUT_PATH = "sim/plotting/sneak_comparison.png"

R_VALUES = [0.0, 10.0, 50.0, 100.0]


def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def main():
    results = load_results()

    rs, coupled_means, decoupled_means = [], [], []
    coupled_mins, coupled_maxs = [], []
    decoupled_mins, decoupled_maxs = [], []

    for r in R_VALUES:
        key = f"r={r:.1f}"
        cell = results.get(key, {})
        coupled = cell.get("coupled", [])
        decoupled = cell.get("decoupled", [])
        if not coupled or not decoupled:
            continue

        rs.append(r)
        coupled_means.append(np.mean(coupled))
        coupled_mins.append(np.min(coupled))
        coupled_maxs.append(np.max(coupled))
        decoupled_means.append(np.mean(decoupled))
        decoupled_mins.append(np.min(decoupled))
        decoupled_maxs.append(np.max(decoupled))

    rs = np.array(rs)
    coupled_means = np.array(coupled_means)
    decoupled_means = np.array(decoupled_means)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- left panel: coupled vs decoupled accuracy ---
    ax1.plot(rs, coupled_means, marker="o", color="#c0392b", label="coupled (real, with sneak paths)")
    ax1.fill_between(rs, coupled_mins, coupled_maxs, color="#c0392b", alpha=0.15)

    ax1.plot(rs, decoupled_means, marker="o", color="#2b6cb0", label="decoupled (sneak paths removed)")
    ax1.fill_between(rs, decoupled_mins, decoupled_maxs, color="#2b6cb0", alpha=0.15)

    ax1.set_xlabel("Row/column parasitic resistance R (\u03a9)")
    ax1.set_ylabel("Inference accuracy")
    ax1.set_title("Coupled vs. decoupled accuracy\n(shaded = min-max range, n=25/point)", fontsize=16)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left")

    # --- right panel: sneak-path cost (the delta) ---
    sneak_cost = decoupled_means - coupled_means
    ax2.bar([str(int(r)) for r in rs], sneak_cost, color="#7c3aed")
    ax2.set_xlabel("Row/column parasitic resistance R (\u03a9)")
    ax2.set_ylabel("Accuracy attributable to sneak-path coupling")
    ax2.set_title("Sneak-path cost\n(decoupled accuracy \u2212 coupled accuracy)", fontsize=16)
    ax2.set_ylim(0, 0.5)
    ax2.grid(alpha=0.3, axis="y")

    for i, v in enumerate(sneak_cost):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center")

    fig.suptitle("Sneak-path current isolation: mixed TEAM/Biolek crossbar", fontsize=22)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()