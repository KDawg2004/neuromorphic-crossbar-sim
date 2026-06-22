import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class BiolekMemcapacitor:
    """
    Biolek memcapacitor, ported from NgSpice .SUBCKT memC.

    Original netlist structure:
      Cq, Gq      -- integrator: dq/dt = i(t)  (q is genuine accumulated charge)
      Cx, Gx      -- integrator: dx/dt = k * q * window(x, i)
      Emc         -- output: v = DM(x) * (q + IC*Cinit)
      DM(x)       -- elastance: 1/Cmax + (1/Cmin - 1/Cmax)*x
      Joglekar window: 1 - (2x-1)^(2p)
      Biolek window:   1 - (x - (1 - sgn(i))/2)^(2p)

    x is clamped to [0, 1] per the SPICE .define Xlimited limiter.

    Testbench drive: q(t) = Q_amp * sin(2*pi*f*t), so i = dq/dt = Q_amp*2*pi*f*cos(...).
    This gives bipolar charge (q oscillates around 0), which is required for
    the state variable x to swing in both directions and produce pinched hysteresis.
    """

    def __init__(self, Cmin=10e-9, Cmax=10e-6, Cinit=100e-9, k=1e7, p=1, IC=0.0,
                 window_type='joglekar'):
        self.Cmin = Cmin
        self.Cmax = Cmax
        self.Cinit = Cinit
        self.k = k
        self.p = p
        self.IC = IC
        self.window_type = window_type  # 'joglekar' or 'biolek'

        # x_init matches the SPICE .param x0
        self.x_init = (1/Cinit - 1/Cmax) / (1/Cmin - 1/Cmax)
        self.q_init = 0.0  # q(t) = Q_amp*sin starts at 0, consistent with Cq uncharged

    # -------------------------
    def DM(self, x):
        """Elastance (inverse capacitance). x is clamped before use."""
        xc = np.clip(x, 0, 1)
        return 1/self.Cmax + (1/self.Cmin - 1/self.Cmax) * xc

    def C(self, x):
        return 1 / self.DM(x)

    def window(self, x, i):
        """
        Window function. Receives clamped x and current i.
        Joglekar: depends only on x.
        Biolek: depends on x and sign(i), which prevents x from leaving [0,1]
                more gracefully than hard clamping alone.
        """
        xc = np.clip(x, 0, 1)
        if self.window_type == 'biolek':
            return 1 - (xc - (1 - np.sign(i)) / 2) ** (2 * self.p)
        else:  # joglekar (default)
            return 1 - (2 * xc - 1) ** (2 * self.p)

    def voltage(self, q, x):
        """Output port equation: v = DM(x) * (q + IC*Cinit)."""
        return self.DM(x) * (q + self.IC * self.Cinit)

    # -------------------------
    def charge_refrence(self, t, freq, Q_amp):
        """Reference charge drive: q(t) = Q_amp * sin(2*pi*f*t)."""
        return Q_amp * np.sin(2 * np.pi * freq * t)
    
    def current_drive(self, t, freq=1.0, Q_amp=100e-9):
        """
        Drive current derived from q(t) = Q_amp * sin(2*pi*f*t).
        i(t) = dq/dt = Q_amp * 2*pi*f * cos(2*pi*f*t).

        This keeps q bipolar (oscillates around 0), which is necessary for x
        to swing in both directions and produce a proper hysteresis loop.

        Q_amp is the peak charge in Coulombs. Choose it so that
        v_peak = DM(x_init) * Q_amp is a reasonable voltage (e.g. ~1V).
        """
        return Q_amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)

    def ode(self, t, y, freq, Q_amp):
        q, x = y
        i = self.current_drive(t, freq, Q_amp)

        dqdt = i                                             # dq/dt = i(t), per Cq/Gq integrator

        x_eff = np.clip(x, 0, 1)
        dxdt = self.k * q * self.window(x_eff, i)           # dx/dt = k*q*window(x,i), per Cx/Gx
        if (x <= 0 and dxdt < 0) or (x >= 1 and dxdt > 0): # stop integrating at boundary
            dxdt = 0.0

        return [dqdt, dxdt]

    # -------------------------
    def simulate(self, t_end=5, freq=1.0, Q_amp=100e-9, n_points=5000):
        t_eval = np.linspace(0, t_end, n_points)

        sol = solve_ivp(
            self.ode,
            [0, t_end],
            [self.q_init, self.x_init],
            t_eval=t_eval,
            args=(freq, Q_amp),
            method="Radau",
            max_step=1/(freq*200),
            rtol=1e-8,
            atol=1e-12,
        )

        t = sol.t
        q = sol.y[0]
        x = sol.y[1]
        v = self.voltage(q, x)
        i = self.current_drive(t, freq, Q_amp)

        return t, q, x, v, i

    def plot(self, **kwargs):
        t, q, x, v, i = self.simulate(**kwargs)

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

        axes[2].plot(t, self.C(x) * 1e9)
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


if __name__ == "__main__":
    # Paper validation params: Cmin=50n, Cmax=200n, Cinit=100n, p=10, Joglekar window
    m = BiolekMemcapacitor(Cmin=50e-9, Cmax=200e-9, Cinit=100e-9, k=1e7, p=10,
                           window_type='joglekar')
    print(f"x_init = {m.x_init:.4f}")
    print(f"C at x_init = {m.C(m.x_init)*1e9:.2f} nF (should be ~100nF)")

    t, q, x, v, i = m.plot(t_end=5.0, freq=1.0, Q_amp=100e-9)
    plt.savefig('biolek_fixed.png', dpi=150, bbox_inches='tight')

    xc = np.clip(x, 0, 1)
    print(f"\nx range: [{xc.min():.4f}, {xc.max():.4f}]")
    print(f"q range: [{q.min()*1e9:.2f}, {q.max()*1e9:.2f}] nC")
    print(f"v range: [{v.min():.4f}, {v.max():.4f}] V")
    print(f"C range: [{m.C(xc.min())*1e9:.2f}, {m.C(xc.max())*1e9:.2f}] nF")

    # Fingerprint check: v and q should cross zero at the same times
    v_zc = t[np.where(np.diff(np.sign(v)))[0]]
    q_zc = t[np.where(np.diff(np.sign(q)))[0]]
    print(f"\nFingerprint check (v and q zero crossings should match):")
    print(f"  v zeros: {v_zc[:6].round(3)}")
    print(f"  q zeros: {q_zc[:6].round(3)}")