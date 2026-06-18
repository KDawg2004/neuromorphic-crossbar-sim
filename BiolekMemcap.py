import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class BiolekMemcapacitor:
    """
    Biolek memcapacitor, ported from NgSpice .SUBCKT memC.

    Original netlist structure:
      Cq, Gq      -- integrator: dq/dt = i(t)  (q is genuine accumulated charge)
      Cx, Gx      -- integrator: dx/dt = k * q * window(x, p)
      Emc         -- output: v = DM(x) * (q + IC*Cinit)
      DM(x)       -- elastance: 1/Cmax + (1/Cmin - 1/Cmax)*x
      window(x,p) -- 1 - (2x-1)^(2p)

    Two coupled state variables: q (charge) and x (internal state).
    This differs from the earlier algebraic-inversion version, where q was
    derived instantaneously from v and x. Here q is a true integrator state,
    matching the SPICE circuit's actual topology (Cq is a real capacitor
    integrating current).
    """

    def __init__(self, Cmin=10e-9, Cmax=10e-6, Cinit=100e-9, k = 1e7, p=1, IC=0.0):
        self.Cmin = Cmin
        self.Cmax = Cmax
        self.Cinit = Cinit
        self.k = k
        self.p = p
        self.IC = IC  # initial charge offset coefficient, per netlist

        # xinit, computed the same way as the SPICE .param
        self.x_init = (1/Cinit - 1/Cmax) / (1/Cmin - 1/Cmax)
        self.q_init = 0.0  # Cq starts uncharged, per netlist (no IC on Cq)

    # -------------------------
    def DM(self, x):
        """Elastance (inverse capacitance), eq. 7 in mentor's reference."""
        return 1/self.Cmax + (1/self.Cmin - 1/self.Cmax) * x

    def C(self, x):
        return 1 / self.DM(x)

    def window(self, x):
        return 1 - (2*x - 1)**(2*self.p)

    def voltage(self, q, x):
        """Output port equation: v = DM(x) * (q + IC*Cinit)."""
        return self.DM(x) * (q + self.IC * self.Cinit)

    # -------------------------
    def current_drive(self, t, freq=1.0, amp=1e-3):
        """
        The netlist's Emc is a voltage-controlled voltage source -- the
        component itself is voltage-output, current-input (like a real
        capacitor: you push current in, voltage appears across it).
        For a clean validation testbench, drive with a current source
        directly, which matches dq/dt = i(t) exactly with no inversion needed.
        """
        return amp * np.sin(2*np.pi*freq*t)

    def ode(self, t, y, freq, amp):
        q, x = y
        i = self.current_drive(t, freq, amp)

        dqdt = -i                                    # Cq/Gq: integrates current directly
        dxdt = self.k * q * self.window(x)           # Cx/Gx: eq. 8, driven by charge

        return [dqdt, dxdt]

    # -------------------------
    def simulate(self, t_end=10, freq=1.0, amp=1e-3, n_points=5000):
        t_eval = np.linspace(0, t_end, n_points)

        sol = solve_ivp(
            self.ode,
            [0, t_end],
            [self.q_init, self.x_init],
            t_eval=t_eval,
            args=(freq, amp),
            method="Radau",
            max_step=1/(freq*200),
            rtol=1e-8,
            atol=1e-12,
        )

        t = sol.t
        q = sol.y[0]
        x = sol.y[1]
        v = self.voltage(q, x)
        i = self.current_drive(t, freq, amp)

        return t, q, x, v, i

    def plot(self, **kwargs):
        t, q, x, v, i = self.simulate(**kwargs)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].plot(v, q)
        axes[0].set_title("q-v loop")
        axes[0].set_xlabel("v (V)")
        axes[0].set_ylabel("q (C)")
        axes[0].grid(alpha=0.3)

        axes[1].plot(t, x)
        axes[1].set_title("state x")
        axes[1].set_xlabel("t (s)")
        axes[1].grid(alpha=0.3)

        axes[2].plot(t, self.C(x) * 1e9)
        axes[2].set_title("capacitance (nF)")
        axes[2].set_xlabel("t (s)")
        axes[2].grid(alpha=0.3)

        axes[3].plot(t, q, label='q')
        axes[3].plot(t, i, label='i (drive)', alpha=0.6)
        axes[3].set_title("q(t) and drive current")
        axes[3].set_xlabel("t (s)")
        axes[3].legend()
        axes[3].grid(alpha=0.3)

        plt.tight_layout()
        return t, q, x, v, i


if __name__ == "__main__":
    m = BiolekMemcapacitor()
    print(f"x_init = {m.x_init:.4f}")
    print(f"C at x_init = {m.C(m.x_init)*1e9:.2f} nF (should be ~100nF)")

    t, q, x, v, i = m.plot(t_end=2.0, freq=1.0, amp=1e-3)
    plt.savefig('biolek_v2.png', dpi=150, bbox_inches='tight')

    print(m.x_init)
    print(m.C(m.x_init))
    print(f"\nx range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"q range: [{q.min()*1e3:.4f}, {q.max()*1e3:.4f}] mC")
    print(f"C range: [{m.C(x.max())*1e9:.2f}, {m.C(x.min())*1e9:.2f}] nF")
    for target in [0.1, 0.5, 1.0]:
        idx = np.argmin(np.abs(t - target))
        print(target, t[idx], x[idx], q[idx])