from devices.endurance import degraded_biolek_defaults
from scipy import stats
import json

with open("sim/training/endurance_sweep_results.json") as f:
    r = json.load(f)

t, p = stats.ttest_ind(r["N=0e+00"], r["N=1e+10"])
print(f"t={t:.3f}, p={p:.4f}")