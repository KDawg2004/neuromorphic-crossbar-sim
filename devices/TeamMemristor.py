#Author: Kaevin Barta
#File/class: Team_memristor/TeamMemristor.py
import numpy as np
from scipy.integrate import solve_ivp
from .memristive import Memristive

class TEAMMemristor(Memristive):
    """
    TEAM (Threshold Adaptive Memristor) model as a reusable class.
    This class implements the TEAM memristor model, which captures the nonlinear dynamics of memristive devices with threshold behavior.
    Contains methods for simulating the device's response to voltage inputs, calculating conductance and resistance, and resetting the state variable.
    """
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
        """
        Initialize the TEAM memristor model with the given parameters.
        k_off: switching rate for off-state\n
        k_on: switching rate for on-state\n
        alpha_off: exponent for off-state switching\n
        alpha_on: exponent for on-state switching\n
        i_off: threshold current for off-state\n
        i_on: threshold current for on-state\n
        G_on: conductance in on-state\n
        G_off: conductance in off-state\n
        w_init: initial state variable (0=off, 1=on)\n
        p: window function exponent\n
        """
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
        self.w = w_init

    def set_state(self, w) -> None:
        """
        Update the initial state variable for the next simulation.\n
        w: new state variable (0=off, 1=on)
        """
        self.w_init = np.clip(w, 0, 1)

    
    def window(self, w, i) -> float:
        """
        Nonlinear window function that reduces the switching rate near the boundaries (w=0 or w=1).\n
        w: state variable (0=off, 1=on)\n
        i: current through the memristor\n
        Returns: window function value"""
        w = np.clip(w, 0.0, 1.0)
        if i >= 0:
            return 1 - w**(2*self.p)
        else:
            return 1 - (1 - w)**(2*self.p)
    
    def conductance(self, w):
        """
        Calculate the conductance of the memristor based on the state variable w.\n
        w: state variable (0=off, 1=on)\n"""
        w = np.clip(w, 0.0, 1.0)
        return self.G_off + w * (self.G_on - self.G_off)
    
    def set_conductance(self, G):
        """
        Program the device to the requested conductance.
        """
        G = np.clip(G, self.G_off, self.G_on)

        self.w = (G - self.G_off) / (self.G_on - self.G_off)
    
    
    def current(self, v):
        """
        Return the instantaneous current through the memristor
        without updating its internal state.
        """
        return self.conductance(self.w) * v
    
    def current_conductance(self, dt):
        return self.conductance(self.w)

    def current_offset(self, dt):
        return 0.0

    def dw_dt(self, w, i):
        """"
        Calculate the rate of change of the state variable w based on the current i.\n
        w: state variable (0=off, 1=on)\n
        i: current through the memristor"""
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
    
    def network_step(self, v, dt):
        self.step(v, dt)

    def network_current(self, v):
        return self.current(v)
    
    def step(self, v, dt) -> float:
        """
        Advance the memristor state by one timestep dt under voltage v.
        Used by the crossbar time loop instead of simulate().
        v: applied voltage (V)
        dt: timestep (s)
        Returns: current through device (A)
        """
        G = self.conductance(self.w_init)
        i = G * v
        dw = self.dw_dt(self.w_init, i)
        self.w_init = np.clip(self.w_init + dt * dw, 0, 1)
        return i

    def state(self) -> float:
        """Return current state variable w."""
        return self.w_init

    def simulate(self, freq=1.0, V_amp=1.5, cycles=3, voltage_fn=None):
        """
        Simulate the behavior of the TEAM memristor under a given voltage input.
        freq: frequency of the voltage source\n
        V_amp: amplitude of the voltage source\n
        cycles: number of cycles to simulate\n
        voltage_fn: custom voltage function (optional)\n
        Returns: time, state variable, voltage, current, and charge arrays
        """
        if voltage_fn is None:
            voltage_source = lambda t: V_amp * np.sin(2 * np.pi * freq * t)
        else:
            voltage_source = voltage_fn

        def ode(t, y):
            """
            ODE function for the memristor state dynamics.\n
            t: time\n
            y: state variable array [w]\n
            Returns: derivative of the state variable [dw/dt]"""
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

    def resistance(self, w) -> float:
        """
        Calculate the resistance of the memristor based on the state variable w.\n
        w: state variable (0=off, 1=on)\n
        Returns: resistance value"""
        return 1 / self.conductance(w)

    def reset(self) -> None:
        """
        Reset the state variable to its initial value for a new simulation."""
        self.w_init = 0.5