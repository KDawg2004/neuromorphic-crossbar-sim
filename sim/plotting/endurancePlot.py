import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = "sim/training/endurance_sweep_results.json"
OUT_PATH = "sim/plotting/endurance_sweep.png"

N_CYCLES = [0, 1e3, 1e4, 1e5, 3e5, 6e5, 1e6, 3e6, 1e7, 1e8, 1e10]


def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def main():
    results = load_results()

    ns, means, mins, maxs = [], [], [], []
    for N in N_CYCLES:
        key = f"N={N:.0e}"
        accs = results.get(key, [])
        if not accs:
            continue
        # N=0 can't be plotted on a log axis; substitute 1 as a stand-in
        # (matches the endurance model's own clamp of log10(max(N,1)))
        ns.append(max(N, 1))
        means.append(np.mean(accs))
        mins.append(np.min(accs))
        maxs.append(np.max(accs))

    ns = np.array(ns)
    means = np.array(means)
    mins = np.array(mins)
    maxs = np.array(maxs)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(ns, means, marker="o", color="#2b6cb0", label="mean accuracy (n=25/trial)")
    ax.fill_between(ns, mins, maxs, color="#2b6cb0", alpha=0.15, label="min-max range")

    ax.axvline(1e5, color="gray", linestyle="--", linewidth=1,
               label="HZO-MS stable-cycling claim (1e5)")
    ax.axhline(0.25, color="lightgray", linestyle=":", linewidth=1,
               label="chance (4-class)")

    ax.set_xscale("log")
    ax.set_xlabel("Write/erase cycle count (N)")
    ax.set_ylabel("Inference accuracy")
    ax.set_title("Mixed TEAM/Biolek crossbar accuracy vs. Biolek endurance degradation\n"
                  "(Biolek window degraded per digitized HZO-MS Fig 3f curve, TEAM held nominal, "
                  "CV=0.10, R=0)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()