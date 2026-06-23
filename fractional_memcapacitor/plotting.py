import numpy as np
import matplotlib.pyplot as plt


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
