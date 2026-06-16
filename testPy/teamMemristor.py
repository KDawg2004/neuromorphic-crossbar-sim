import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# TEAM parameters
# ============================================================

PARAMS = {

    # switching rates
    'k_off': 1,
    'k_on': -1,

    # nonlinearity
    'alpha_off': 5,
    'alpha_on': 5,

    # threshold currents
    'i_off': 0.5e-3,
    'i_on': -0.5e-3,

    # conductance limits
    'G_on': 1/1e3,      # 1 kOhm
    'G_off': 1/10e3,    # 10 kOhm

    # initial state
    'w_init': 0.5,

    # window exponent
    'p': 2
}


# ============================================================
# Window function
# ============================================================

def window(w, i, p):

    w = np.clip(w, 0.0, 1.0)

    if i >= 0:
        return 1 - w**(2*p)
    else:
        return 1 - (1-w)**(2*p)


# ============================================================
# Conductance model
# ============================================================

def conductance(w, p):

    w = np.clip(w, 0.0, 1.0)

    return (
        p['G_off']
        + w*(p['G_on'] - p['G_off'])
    )


# ============================================================
# TEAM state equation
# ============================================================

def dw_dt(w, i, p):

    w = np.clip(w, 0.0, 1.0)

    if i >= p['i_off']:

        dw = (
            p['k_off']
            * ((i/p['i_off']) - 1)**p['alpha_off']
            * window(w, i, p['p'])
        )

    elif i <= p['i_on']:

        dw = (
            p['k_on']
            * (((-i)/abs(p['i_on'])) - 1)**p['alpha_on']
            * window(w, i, p['p'])
        )

    else:
        dw = 0.0

    # hard bounds

    if w <= 0 and dw < 0:
        dw = 0

    if w >= 1 and dw > 0:
        dw = 0

    return dw


# ============================================================
# Simulation
# ============================================================

def simulate(freq=1, V_amp=2.0, cycles=3):

    p = PARAMS

    def voltage(t):
        return V_amp*np.sin(2*np.pi*freq*t)

    def ode(t, y):

        w = y[0]

        v = voltage(t)

        G = conductance(w, p)

        i = G*v

        return [dw_dt(w, i, p)]

    T = 1/freq

    t_end = cycles*T

    t_eval = np.linspace(0, t_end, 10000)

    sol = solve_ivp(
        ode,
        [0, t_end],
        [p['w_init']],
        t_eval=t_eval,
        method='RK45',
        max_step=T/1000,
        rtol=1e-8,
        atol=1e-10
    )

    t = sol.t

    w = np.clip(sol.y[0], 0, 1)

    v = voltage(t)

    G = conductance(w, p)

    i = G*v

    dw_values = []

    for wi, vi in zip(w, v):
        ii = conductance(wi, PARAMS) * vi
        dw_values.append(dw_dt(wi, ii, PARAMS))

    print("max |dw/dt| =", np.max(np.abs(dw_values)))


    return t, w, v, i


# ============================================================
# Plotting
# ============================================================

def plot_results(freq):

    t, w, v, i = simulate(freq=freq)

    fig, ax = plt.subplots(1,3, figsize=(15,4))

    # I-V loop

    ax[0].plot(v, i*1e3)

    ax[0].set_xlabel('Voltage (V)')
    ax[0].set_ylabel('Current (mA)')
    ax[0].set_title(f'I-V Hysteresis ({freq} Hz)')
    ax[0].grid()

    # state variable

    ax[1].plot(t, w)

    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('w')
    ax[1].set_title('State Variable')
    ax[1].grid()

    # conductance

    G = conductance(w, PARAMS)

    ax[2].plot(t, G*1e3)

    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Conductance (mS)')
    ax[2].set_title('Conductance')
    ax[2].grid()

    print(f"{freq} Hz: w in [{w.min():.4f}, {w.max():.4f}]")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    plot_results(freq=0.1)
    plot_results(freq=1)
    plot_results(freq=10)
    plot_results(freq=100)