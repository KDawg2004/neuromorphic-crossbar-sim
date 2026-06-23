import numpy as np

from .. import TEAMMemristor, plot


if __name__ == "__main__":
    m = TEAMMemristor()
    print(f"w_init = {m.w_init:.4f}")
    print(f"G at w_init = {m.conductance(m.w_init)*1e3:.2f} mS")
    print(f"R at w_init = {m.resistance(m.w_init)/1e3:.2f} kOhm (should be ~5.5 kOhm)")

    t, w, v, i, q = plot(m, t_end=3.0, freq=0.5, amp=2.0)

    print(f"\nw range: [{w.min():.4f}, {w.max():.4f}]")
    print(f"q range: [{q.min()*1e3:.4f}, {q.max()*1e3:.4f}] mC")
    print(f"v range: [{v.min():.4f}, {v.max():.4f}] V")
    print(f"G range: [{m.conductance(w.max())*1e3:.2f}, {m.conductance(w.min())*1e3:.2f}] mS")

    for target in [0.5, 1.0, 1.5, 2.0]:
        idx = np.argmin(np.abs(t - target))
        print(f"t={target:1.1f}s -> t={t[idx]:.4f}s, w={w[idx]:.4f}, q={q[idx]*1e3:.6f} mC, v={v[idx]:.4f} V")