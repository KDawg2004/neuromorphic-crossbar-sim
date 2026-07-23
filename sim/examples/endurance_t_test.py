from scipy import stats
import json

with open("sim/training/endurance_sweep_results.json") as f:
    r = json.load(f)

for key in ["N=1e+07", "N=1e+08", "N=1e+10"]:
    t, p = stats.ttest_ind(r["N=0e+00"], r[key])
    print(f"N=0 vs {key}: t={t:.3f}, p={p:.4f}")