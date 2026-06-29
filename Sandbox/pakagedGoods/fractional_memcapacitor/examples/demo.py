import numpy as np
import matplotlib.pyplot as plt

from ..model import BiolekMemcapacitor


if __name__ == "__main__":

    m = BiolekMemcapacitor(
        Cmin=50e-9,
        Cmax=200e-9,
        Cinit=100e-9,
        k=1e7,
        p=10,
        window_type='joglekar'
    )

    print(f"x_init = {m.x_init:.4f}")
    print(f"C at x_init = {m.C(m.x_init)*1e9:.2f} nF (should be ~100nF)")

    # -------------------------------------------------
    # Reference solution using existing alpha=1 solver
    # -------------------------------------------------

    t, q_ref, x_ref, v_ref, i_ref = m.simulate(
        t_end=5,
        freq=1.0,
        Q_amp=100e-9,
        alpha=1.0,
        n_points=10000
    )

    # -------------------------------------------------
    # New timestep implementation
    # -------------------------------------------------

    m_step = BiolekMemcapacitor(
        Cmin=50e-9,
        Cmax=200e-9,
        Cinit=100e-9,
        k=1e7,
        p=10,
        window_type='joglekar'
    )

    dt = t[1] - t[0]

    x_step = np.zeros_like(t)
    q_step = np.zeros_like(t)
    v_step = np.zeros_like(t)

    for n in range(len(t)):

        x_step[n] = m_step.x
        q_step[n] = m_step.q
        v_step[n] = m_step.voltage(m_step.q, m_step.x)

        m_step.step(i_ref[n], dt)

    # -------------------------------------------------
    # Compare state trajectories
    # -------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(t, x_ref, label="simulate(alpha=1)")
    plt.plot(t, x_step, "--", label="step()")
    plt.xlabel("Time (s)")
    plt.ylabel("State x")
    plt.title("Alpha=1 vs step() Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------------------------
    # Compare Q-V hysteresis
    # -------------------------------------------------

    plt.figure(figsize=(6, 6))
    plt.plot(v_ref, q_ref * 1e9, label="simulate(alpha=1)")
    plt.plot(v_step, q_step * 1e9, "--", label="step()")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Charge (nC)")
    plt.title("Q-V Hysteresis Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------------------------
    # Numerical comparison
    # -------------------------------------------------
    print("\nReference x:")
    print(x_ref[:10])

    print("\nDifference:")
    print((x_ref - x_step)[:10])

    print("\nStep x:")
    print(x_step[:10])
    print("\nMaximum state difference:")
    print(np.max(np.abs(x_ref - x_step)))
    print("Reference q max:", np.max(q_ref))
    print("Step q max:", np.max(q_step))

    print("Reference q min:", np.min(q_ref))
    print("Step q min:", np.min(q_step))

"""import matplotlib.pyplot as plt

from ..model import BiolekMemcapacitor
from ..plotting import plot_model
from ..validation import validate_alpha_one, validate_alpha_sweep


if __name__ == "__main__":
    # Paper validation params: Cmin=50n, Cmax=200n, Cinit=100n, p=10, Joglekar window
    m = BiolekMemcapacitor(Cmin=50e-9, Cmax=200e-9, Cinit=100e-9, k=1e7, p=10,
                           window_type='joglekar')
    print(f"x_init = {m.x_init:.4f}")
    print(f"C at x_init = {m.C(m.x_init)*1e9:.2f} nF (should be ~100nF)")

    t, q, x, v, i = validate_alpha_one(m)

    plt.figure(figsize=(6,6))
    plt.plot(v, q*1e9)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Charge (nC)")
    plt.title("Alpha = 1 Validation")
    plt.grid(True)
    plt.show()
"""
    
