import numpy as np
from math import gamma


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
    def simulate(self, t_end=5, freq=1.0, Q_amp=100e-9, alpha=1.0, n_points=1000):
        t_eval = np.linspace(0, t_end, n_points)

        t = np.linspace(0, t_end, n_points)

        q = self.charge_refrence(t, freq, Q_amp)
        i = self.current_drive(t, freq, Q_amp)

        x = self.fractional_state(t, q, i, alpha)

        v = self.voltage(q, x)

        return t, q, x, v, i

    def fractional_state(self, t, q, i, alpha):

        # alpha=1 should recover the ordinary ODE.
        if abs(alpha - 1.0) < 1e-12:

            x = np.zeros_like(t)
            x[0] = self.x_init

            dt = t[1] - t[0]

            for n in range(1, len(t)):

                xeff = np.clip(x[n-1], 0.0, 1.0)

                rhs = self.k * q[n-1] * self.window(xeff, i[n-1])

                x[n] = x[n-1] + dt * rhs

                x[n] = np.clip(x[n], 0.0, 1.0)

            return x

        N = len(t)
        h = t[1] - t[0]

        x = np.zeros(N)
        x[0] = self.x_init

        f = np.zeros(N)

        f[0] = self.k * q[0] * self.window(np.clip(x[0], 0, 1), i[0])

        g1 = gamma(alpha + 1)
        g2 = gamma(alpha + 2)

        for n in range(N - 1):

            predictor = x[0]

            for j in range(n + 1):

                b = (n + 1 - j)**alpha - (n - j)**alpha

                predictor += h**alpha / g1 * b * f[j]

            xp = np.clip(predictor, 0.0, 1.0)

            fp = self.k * q[n + 1] * self.window(xp, i[n + 1])

            corrector = x[0]

            for j in range(n + 1):

                if j == 0:

                    a = n**(alpha + 1) - (n - alpha) * (n + 1)**alpha

                else:

                    k = n - j + 1

                    a = (k + 1)**(alpha + 1) - 2 * k**(alpha + 1) + (k - 1)**(alpha + 1)

                corrector += h**alpha / g2 * a * f[j]

            corrector += h**alpha / g2 * fp

            x[n + 1] = np.clip(corrector, 0.0, 1.0)

            f[n + 1] = self.k * q[n + 1] * self.window(
                np.clip(x[n + 1], 0.0, 1.0),
                i[n + 1]
            )

        return x


