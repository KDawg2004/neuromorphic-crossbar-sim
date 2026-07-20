import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = "sim/training/parasitic_variability_grid_results.json"
OUT_PATH = "sim/plotting/parasitic_variability_grid.png"

CV_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]
R_VALUES = [0.0, 10.0, 50.0, 100.0]


def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def main():
    results = load_results()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(R_VALUES)))

    for r, color in zip(R_VALUES, colors):
        means, mins, maxs = [], [], []
        for cv in CV_VALUES:
            key = f"cv={cv:.2f}_r={r:.1f}"
            accs = results.get(key, [])
            if not accs:
                means.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)
                continue
            means.append(np.mean(accs))
            mins.append(np.min(accs))
            maxs.append(np.max(accs))

        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)

        ax.plot(CV_VALUES, means, marker="o", color=color, label=f"R = {r:.0f} \u03a9")
        ax.fill_between(CV_VALUES, mins, maxs, color=color, alpha=0.15)

    ax.set_xlabel("Device-to-device variability (CV)")
    ax.set_ylabel("Inference accuracy")
    ax.set_title("Mixed-device crossbar accuracy vs. variability and parasitic resistance\n"
                  "(lines = mean across trials, shaded band = min-max range, n=25 per point)")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="chance (4-class)")
    ax.legend(title="Row/column resistance", loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()