import numpy as np
from .. import TEAMMemristor, plot


m = TEAMMemristor(
    k_off=1.333,
    k_on=-1.333,
    alpha_off=2,
    alpha_on=2,
    i_off=0.5e-3,
    i_on=-0.5e-3,
    G_on=1/500,
    G_off=1/5000,
    w_init=0.5,
)

# Peak current at this amplitude = G_on * V_amp = (1/500) * 0.1 = 0.2mA
# i_off = 0.5mA, so we never exceed threshold -> no switching -> straight line
t, w, v, i, q = plot(m, t_end=3.0, freq=1.0, amp=0.9)
print(f"w range: [{w.min():.6f}, {w.max():.6f}]")  # should be essentially flat