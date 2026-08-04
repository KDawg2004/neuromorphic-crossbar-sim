from scipy import stats
import json

with open("sim/training/endurance_sweep_results.json") as f:
    r = json.load(f)

baseline = r["N=0e+00"]

for key in ["N=1e+03", "N=1e+05", "N=1e+07", "N=1e+08", "N=1e+10"]:
    t, p = stats.ttest_ind(baseline, r[key])
    direction = "higher" if sum(r[key])/len(r[key]) > sum(baseline)/len(baseline) else "lower"
    print(f"N=0 vs {key}: t={t:.3f}, p={p:.4f}  ({direction} than baseline)")