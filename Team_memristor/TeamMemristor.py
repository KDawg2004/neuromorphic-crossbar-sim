import numpy as np
from scipy.integrate import solve_ivp

class TEAMMemristor:

    def __init__(
        self,
        k_off=1,
        k_on=-1,
        alpha_off=5,
        alpha_on=5,
        i_off=0.5e-3,
        i_on=-0.5e-3,
        G_on=1/1e3,
        G_off=1/10e3,
        w_init=0.5,
        p=2,
    ):
        self.k_off = k_off
        self.k_on = k_on
        self.alpha_off = alpha_off
        self.alpha_on = alpha_on
        self.i_off = i_off
        self.i_on = i_on
        self.G_on = G_on
        self.G_off = G_off
        self.w_init = w_init
        self.p = p

    def set_state(self, w):
        self.w_init = np.clip(w, 0, 1)

    def window(self, w, i):
        # Biolek window: direction-dependent, prevents boundary lock-in
        w = np.clip(w, 0.0, 1.0)
        if i >= 0:
            return 1 - w**(2*self.p)
        else:
            return 1 - (1 - w)**(2*self.p)

    def conductance(self, w):
        # Linear interpolation between G_off and G_on
        w = np.clip(w, 0.0, 1.0)
        return self.G_off + w * (self.G_on - self.G_off)

    def dw_dt(self, w, i):
        w = np.clip(w, 0.0, 1.0)

        if i >= self.i_off:
            dw = self.k_off * ((i / self.i_off) - 1)**self.alpha_off * self.window(w, i)
        elif i <= self.i_on:
            dw = self.k_on * ((i / self.i_on) - 1)**self.alpha_on * self.window(w, i)
        else:
            dw = 0.0

        if w >= 1.0 and dw > 0:
            dw = 0.0
        if w <= 0.0 and dw < 0:
            dw = 0.0

        return dw

    def simulate(self, freq=1.0, V_amp=1.5, cycles=3, voltage_fn=None):
        if voltage_fn is None:
            voltage_source = lambda t: V_amp * np.sin(2 * np.pi * freq * t)
        else:
            voltage_source = voltage_fn

        def ode(t, y):
            w = y[0]
            v = voltage_source(t)
            G = self.conductance(w)
            i = G * v
            return [self.dw_dt(w, i)]

        T = 1 / freq
        t_end = cycles * T
        t_eval = np.linspace(0, t_end, 10000)

        sol = solve_ivp(ode, [0, t_end], [self.w_init], t_eval=t_eval,
                        method='RK45', max_step=T/1000, rtol=1e-8, atol=1e-10)

        w = np.clip(sol.y[0], 0, 1)
        t = sol.t
        v = voltage_source(t)
        G = self.conductance(w)
        i = G * v
        q = np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))
        q = np.insert(q, 0, 0)

        return t, w, v, i, q

    def resistance(self, w):
        return 1 / self.conductance(w)

    def reset(self):
        self.w_init = 0.5