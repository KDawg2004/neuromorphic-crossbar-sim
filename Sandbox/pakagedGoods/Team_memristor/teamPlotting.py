import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import plasma
from matplotlib.collections import LineCollection


# Dark theme palette
_BG       = '#0d0d0f'
_PANEL    = '#13131a'
_BORDER   = '#2a2a3d'
_TEXT     = '#e8e8f0'
_SUBTEXT  = '#7070a0'
_VOLTAGE  = '#7eb8f7'   # cool blue
_CURRENT  = '#f7a84a'   # warm amber
_IV       = '#c084fc'   # violet
_QV       = '#34d399'   # teal
_STATE    = '#fb7185'   # rose


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(_SUBTEXT)
    ax.yaxis.label.set_color(_SUBTEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.grid(True, color=_BORDER, linewidth=0.5, linestyle='--', alpha=0.6)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=9, fontweight='bold', pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def plot(model, t_end=2.0, freq=1.0, amp=1.5, voltage_fn=None):
    """
    Plot TEAM memristor behavior.

    Parameters
    ----------
    model      : TEAMMemristor instance
    t_end      : simulation end time (s)
    freq       : excitation frequency (Hz)
    amp        : voltage amplitude (V)
    voltage_fn : optional callable v(t); overrides amp/freq sine wave

    Returns
    -------
    t, w, v, i, q
    """
    cycles = max(1, int(t_end * freq))
    t, w, v, i, q = model.simulate(
        freq=freq, V_amp=amp, cycles=cycles, voltage_fn=voltage_fn
    )

    fig = plt.figure(figsize=(12, 9), facecolor=_BG)
    fig.suptitle(
        'TEAM Memristor',
        color=_TEXT, fontsize=15, fontweight='bold', y=0.97
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.38, wspace=0.32,
        left=0.08, right=0.96, top=0.91, bottom=0.08
    )

    # --- v(t) and i(t) ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, v,       color=_VOLTAGE, lw=1.4, label='v(t)  [V]')
    ax0.plot(t, i * 1e3, color=_CURRENT, lw=1.4, label='i(t)  [mA]', alpha=0.9)
    _style_ax(ax0, title='Signals', xlabel='Time (s)', ylabel='V  /  mA')
    ax0.legend(
        fontsize=7.5, framealpha=0.15,
        labelcolor=_TEXT, facecolor=_PANEL, edgecolor=_BORDER
    )

    # --- I-V loop, colored by time ---
    ax1 = fig.add_subplot(gs[0, 1])
    points  = np.array([v, i * 1e3]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap='plasma', linewidth=1.3, alpha=0.9)
    lc.set_array(t[:-1])
    ax1.add_collection(lc)
    ax1.autoscale()
    cbar = fig.colorbar(lc, ax=ax1, pad=0.02)
    cbar.set_label('time (s)', color=_SUBTEXT, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=_SUBTEXT, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_SUBTEXT)
    cbar.outline.set_edgecolor(_BORDER)
    _style_ax(ax1, title='I–V Loop', xlabel='Voltage (V)', ylabel='Current (mA)')

    # --- Q-V loop ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(v, q * 1e6, color=_QV, lw=1.4)
    _style_ax(ax2, title='Q–V Loop', xlabel='Voltage (V)', ylabel='Charge (µC)')

    # --- w(t) with gradient fill ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, w, color=_STATE, lw=1.4)
    ax3.fill_between(t, w, alpha=0.12, color=_STATE)
    ax3.axhline(0, color=_BORDER, lw=0.8)
    ax3.axhline(1, color=_BORDER, lw=0.8)
    ax3.set_ylim(-0.05, 1.05)
    _style_ax(ax3, title='State Variable w(t)', xlabel='Time (s)', ylabel='w')

    plt.savefig('team_qv.png', dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.show()

    return t, w, v, i, q