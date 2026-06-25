import numpy as np
import matplotlib.pyplot as plt


def plot(model, t_end=2.0, freq=1.0, amp=1.5, voltage_fn=None):
    """
    Plot TEAM memristor behavior.

    Parameters
    ----------
    model : TEAMMemristor instance
    t_end : simulation end time (s)
    freq  : excitation frequency (Hz)
    amp   : voltage amplitude (V)

    Returns
    -------
    t, w, v, i, q  -- same order as simulate()
    """
    cycles = max(1, int(t_end * freq))

    t, w, v, i, q = model.simulate(
        freq=freq,
        V_amp=amp,
        cycles=cycles,
        voltage_fn=voltage_fn
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('TEAM Memristor', fontsize=14)

    # v(t) and i(t)
    axes[0, 0].plot(t, v, label='v(t)')
    axes[0, 0].plot(t, i * 1e3, label='i(t) (mA)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Voltage / Current')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # I-V loop
    axes[0, 1].plot(v, i * 1e3)
    axes[0, 1].set_xlabel('Voltage (V)')
    axes[0, 1].set_ylabel('Current (mA)')
    axes[0, 1].set_title('I-V loop')
    axes[0, 1].grid(True)

    # Q-V loop
    axes[1, 0].plot(v, q * 1e6, color='red')
    axes[1, 0].set_xlabel('Voltage (V)')
    axes[1, 0].set_ylabel('Charge (uC)')
    axes[1, 0].set_title('Q-V loop')
    axes[1, 0].grid(True)

    # w(t)
    axes[1, 1].plot(t, w)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('State w')
    axes[1, 1].set_title('State variable w(t)')
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('team_qv.png', dpi=150, bbox_inches='tight')
    plt.show()

    return t, w, v, i, q