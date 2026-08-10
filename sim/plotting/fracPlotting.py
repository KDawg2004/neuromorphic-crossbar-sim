import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import plasma

def plot_model(model, **kwargs):
    t, q, x, v, i = model.simulate(**kwargs)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(v, q * 1e9)
    axes[0].set_title("q-v loop (pinched hysteresis)")
    axes[0].set_xlabel("v (V)")
    axes[0].set_ylabel("q (nC)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, np.clip(x, 0, 1))
    axes[1].set_title("state x")
    axes[1].set_xlabel("t (s)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, model.C(x) * 1e9)
    axes[2].set_title("capacitance (nF)")
    axes[2].set_xlabel("t (s)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, q * 1e9, label="q (nC)")
    axes[3].plot(t, i * 1e6, label="i (uA)", alpha=0.6)
    axes[3].set_title("q(t) and drive current")
    axes[3].set_xlabel("t (s)")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    return t, q, x, v, i

def plot_validation_suiteV0(model, t_end=5.0, freq=1.0, Q_amp=100e-9, n_points=1000):
    """
    6-panel validation suite for the fractional-order memcapacitor.
 
    Panels:
      1. x(t) and C(t) on dual y-axis for each alpha
      2. Pinched q-v hysteresis loops (steady state)
      3. Hysteresis loop area vs alpha sweep
      4. C(t) colored by x value at alpha=0.5
      5. Frequency sweep at alpha=0.5
      6. Phase portrait x vs q
    """
    alphas = [1.0, 0.80, 0.60, 0.50, 0.40]
    colors = plasma(np.linspace(0.15, 0.9, len(alphas)))
 
    # pre-simulate all alphas up front
    sims = {a: model.simulate(t_end=t_end, freq=freq, Q_amp=Q_amp,
                              alpha=a, n_points=n_points) for a in alphas}
 
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0f0f1a')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
 
    def style(ax):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_color('#444')
 
    def label(ax, xl, yl, title):
        ax.set_xlabel(xl, color='white')
        ax.set_ylabel(yl, color='white')
        ax.set_title(title, color='white', fontsize=10)
 
    # ── 1: x(t) and C(t) dual-axis ──
    ax1  = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    style(ax1)
    for a, col in zip(alphas, colors):
        t, q, x, v, i = sims[a]
        ax1.plot(t, np.clip(x, 0, 1), color=col, lw=1.6, label=f'α={a}')
        ax1b.plot(t, model.C(x) * 1e9, color=col, lw=1.0, ls='--', alpha=0.45)
    ax1b.set_ylabel('C (nF)', color='#aaa')
    ax1b.tick_params(colors='#aaa')
    label(ax1, 't (s)', 'state x', 'x(t) & C(t)  [solid=x, dashed=C]')
    ax1.legend(fontsize=7, loc='upper right',
               facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)
 
    # ── 2: pinched hysteresis q-v ──
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2)
    for a, col in zip(alphas, colors):
        t, q, x, v, i = sims[a]
        idx = t >= t_end - 2.0
        ax2.plot(v[idx], q[idx] * 1e9, color=col, lw=1.6, label=f'α={a}')
    ax2.axhline(0, color='#555', lw=0.5)
    ax2.axvline(0, color='#555', lw=0.5)
    label(ax2, 'v (V)', 'q (nC)', 'q-v Pinched Hysteresis')
    ax2.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)
 
    # ── 3: loop area vs alpha sweep ──
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3)
    # dense around the peak so 0.5 shows clearly instead of flattening out
    alpha_sweep = np.unique(np.concatenate([
        np.linspace(0.30, 0.45, 6),
        np.linspace(0.45, 0.60, 10),
        np.linspace(0.60, 1.20, 15),
    ]))
    areas = []
    for a in alpha_sweep:
        t, q, x, v, i = model.simulate(t_end=t_end, freq=freq, Q_amp=Q_amp,
                                        alpha=a, n_points=n_points)
        def _poly_area(px, py):
            return 0.5 * abs(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
        area = _poly_area(v[q >= 0], q[q >= 0]) + _poly_area(v[q <= 0], q[q <= 0])
        areas.append(area)
    areas = np.array(areas)
    ax3.scatter(alpha_sweep, areas * 1e9, c=areas, cmap='plasma', s=45, zorder=3)
    ax3.plot(alpha_sweep, areas * 1e9, color='#aaa', lw=1.5, alpha=0.7)
    peak_idx = np.argmax(areas)
    ax3.axvline(alpha_sweep[peak_idx], color='yellow', lw=1, ls='--', alpha=0.8,
                label=f'peak α≈{alpha_sweep[peak_idx]:.2f}')
    ax3.axvline(1.0, color='cyan', lw=1, ls='--', alpha=0.7, label='α=1')
    label(ax3, 'α', 'loop area (nC·V)', 'Hysteresis Area vs α')
    ax3.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)
 
    # ── 4: C(t) colored by x at alpha=0.5 ──
    ax4 = fig.add_subplot(gs[1, 0])
    style(ax4)
    t, q, x, v, i = sims[0.50]
    sc = ax4.scatter(t, model.C(x) * 1e9, c=np.clip(x, 0, 1),
                     cmap='plasma', s=4, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax4)
    cb.set_label('x state', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    label(ax4, 't (s)', 'C (nF)', 'C(t) colored by x  [α=0.5]')
 
    # ── 5: frequency sweep at alpha=0.5 ──
    ax5 = fig.add_subplot(gs[1, 1])
    style(ax5)
    freqs   = [0.5, 1.0, 2.0, 5.0]
    fcolors = plasma(np.linspace(0.15, 0.9, len(freqs)))
    for f, col in zip(freqs, fcolors):
        te = 5 / f
        t2, q2, x2, v2, i2 = model.simulate(t_end=te, freq=f, Q_amp=Q_amp,
                                              alpha=0.5, n_points=n_points)
        idx = t2 >= te - 2 / f
        ax5.plot(v2[idx], q2[idx] * 1e9, color=col, lw=1.6, label=f'{f} Hz')
    ax5.axhline(0, color='#555', lw=0.5)
    ax5.axvline(0, color='#555', lw=0.5)
    label(ax5, 'v (V)', 'q (nC)', 'Freq sweep  [α=0.5]')
    ax5.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)
 
    # ── 6: phase portrait x vs q ──
    ax6 = fig.add_subplot(gs[1, 2])
    style(ax6)
    for a, col in zip(alphas, colors):
        t, q, x, v, i = sims[a]
        idx = t >= t_end - 2.0
        ax6.plot(q[idx] * 1e9, np.clip(x[idx], 0, 1),
                 color=col, lw=1.4, label=f'α={a}')
    label(ax6, 'q (nC)', 'x (state)', 'Phase portrait: x vs q')
    ax6.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)
 
    fig.suptitle('Fractional-Order Biolek Memcapacitor — Validation Suite',
                 color='white', fontsize=18, fontweight='bold', y=0.99)
    plt.show()

def plot_validation_suite(model, t_end=5.0, freq=1.0, Q_amp=100e-9, n_points=1000):
    """
    6-panel validation suite for the fractional-order memcapacitor.

    Panels:
      1. x(t) and C(t) on dual y-axis for each alpha
      2. Pinched q-v hysteresis loops (steady state)
      3. Hysteresis loop area vs alpha sweep (0.1 to 1.2)
      4. C(t) colored by x value at alpha=0.5
      5. Frequency sweep at alpha=0.5
      6. Phase portrait x vs q
    """
    alphas = [1.0, 0.75, 0.50, 0.25, 0.10]
    colors = plasma(np.linspace(0.15, 0.9, len(alphas)))

    sims = {a: model.simulate(t_end=t_end, freq=freq, Q_amp=Q_amp,
                              alpha=a, n_points=n_points) for a in alphas}

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0f0f1a')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    def style(ax):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_color('#444')

    def label(ax, xl, yl, title):
        ax.set_xlabel(xl, color='white')
        ax.set_ylabel(yl, color='white')
        ax.set_title(title, color='white', fontsize=10)

    # ── 1: x(t) and C(t) dual-axis ──
    ax1  = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    style(ax1)
    for a, col in zip(alphas, colors):
        t, q, x, v, i = sims[a]
        ax1.plot(t, np.clip(x, 0, 1), color=col, lw=1.6, label=f'α={a}')
        ax1b.plot(t, model.C(x) * 1e9, color=col, lw=1.0, ls='--', alpha=0.45)
    ax1b.set_ylabel('C (nF)', color='#aaa')
    ax1b.tick_params(colors='#aaa')
    label(ax1, 't (s)', 'state x', 'x(t) & C(t)  [solid=x, dashed=C]')
    ax1.legend(fontsize=7, loc='upper right',
               facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)

    # ── 2: pinched hysteresis q-v ──
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2)
    for a, col in zip(alphas, colors):
        t, q, x, v, i = sims[a]
        idx = t >= t_end - 2.0
        ax2.plot(v[idx], q[idx] * 1e9, color=col, lw=1.6, label=f'α={a}')
    ax2.axhline(0, color='#555', lw=0.5)
    ax2.axvline(0, color='#555', lw=0.5)
    label(ax2, 'v (V)', 'q (nC)', 'q-v Pinched Hysteresis')
    ax2.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)

    # ── 3: loop area vs alpha sweep 0.1 → 1.2 ──
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3)
    alpha_sweep = np.unique(np.concatenate([
        np.linspace(0.10, 0.30, 8),   # fine resolution at low alpha
        np.linspace(0.30, 0.60, 12),  # dense around the expected peak
        np.linspace(0.60, 1.20, 10),  # coarser above
    ]))
    areas = []
    for a in alpha_sweep:
        t, q, x, v, i = model.simulate(t_end=t_end, freq=freq, Q_amp=Q_amp,
                                        alpha=a, n_points=n_points)
        def _poly_area(px, py):
            return 0.5 * abs(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
        # use only steady-state portion
        area = _poly_area(v[q >= 0], q[q >= 0]) + _poly_area(v[q <= 0], q[q <= 0])
        areas.append(area)
    areas = np.array(areas)
    ax3.scatter(alpha_sweep, areas * 1e9, c=areas, cmap='plasma', s=45, zorder=3)
    ax3.plot(alpha_sweep, areas * 1e9, color='#aaa', lw=1.5, alpha=0.7)
    peak_idx = np.argmax(areas)
    ax3.axvline(alpha_sweep[peak_idx], color='yellow', lw=1, ls='--', alpha=0.8,
                label=f'peak α≈{alpha_sweep[peak_idx]:.2f}')
    ax3.axvline(1.0, color='cyan', lw=1, ls='--', alpha=0.7, label='α=1')
    ax3.set_xlim(0.05, 1.25)
    label(ax3, 'α', 'loop area (nC·V)', 'Hysteresis Area vs α  [0.1→1.2]')
    ax3.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)

    # ── 4: C(t) colored by x at alpha=0.5 ──
    ax4 = fig.add_subplot(gs[1, 0])
    style(ax4)
    t, q, x, v, i = sims[0.50]
    sc = ax4.scatter(t, model.C(x) * 1e9, c=np.clip(x, 0, 1),
                     cmap='plasma', s=4, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax4)
    cb.set_label('x state', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    label(ax4, 't (s)', 'C (nF)', 'C(t) colored by x  [α=0.5]')

    # ── 5: frequency sweep at alpha=0.5 ──
    ax5 = fig.add_subplot(gs[1, 1])
    style(ax5)
    freqs   = [0.5, 1.0, 2.0, 5.0]
    fcolors = plasma(np.linspace(0.15, 0.9, len(freqs)))
    for f, col in zip(freqs, fcolors):
        te = 5 / f
        t2, q2, x2, v2, i2 = model.simulate(t_end=te, freq=f, Q_amp=Q_amp,
                                              alpha=0.5, n_points=n_points)
        idx = t2 >= te - 2 / f
        ax5.plot(v2[idx], q2[idx] * 1e9, color=col, lw=1.6, label=f'{f} Hz')
    ax5.axhline(0, color='#555', lw=0.5)
    ax5.axvline(0, color='#555', lw=0.5)
    label(ax5, 'v (V)', 'q (nC)', 'Freq sweep  [α=0.5]')
    ax5.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)

    # ── 6: phase portrait x vs q ──
    ax6 = fig.add_subplot(gs[1, 2])
    style(ax6)
    for a, col in zip(alphas, colors):
        t, q, x, v, i = sims[a]
        idx = t >= t_end - 2.0
        ax6.plot(q[idx] * 1e9, np.clip(x[idx], 0, 1),
                 color=col, lw=1.4, label=f'α={a}')
    label(ax6, 'q (nC)', 'x (state)', 'Phase portrait: x vs q')
    ax6.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', framealpha=0.6)

    fig.suptitle('Fractional-Order Biolek Memcapacitor — Validation Suite',
                 color='white', fontsize=18, fontweight='bold', y=0.99)

    plt.show()