*TEAM Memristor Package*
Team_memristor/TeamMemristor.py provides a reusable Python implementation of the TEAM (Threshold Adaptive Memristor) model. It simulates threshold-based switching under an applied voltage source and returns the key state and electrical waveforms. The model uses solve_ivp from SciPy and a simple plotting helper for quick visualization.

*Installation*
-Requirements
Python 3
NumPy
SciPy
Matplotlib

*Example*
bash
pip install numpy scipy matplotlib
Basic Usage
python
from Team_memristor.TeamMemristor import TEAMMemristor, plot

m = TEAMMemristor(
    k_off=1.333,
    k_on=-1.333,
    alpha_off=2,
    alpha_on=2,
    i_off=0.5e-3,
    i_on=-0.5e-3,
    G_on=1/500,
    G_off=1/5000,
    w_init=0.5,
)

t, w, v, i, q = plot(m, t_end=3.0, freq=1.0, amp=0.9)

*What It Returns*
simulate() and plot() provide:

t: time array
w: internal state variable
v: applied voltage
i: current through the device
q: accumulated charge

*Main Parameters*
i_off, i_on: current thresholds for switching.

k_off, k_on: switching speed beyond threshold.

alpha_off, alpha_on: nonlinearity of switching.

G_on, G_off: conductance values for the two states.

w_init: starting state.

p: window sharpness near boundaries.

*Key Behavior*
No switching occurs when current stays between i_on and i_off.

w is clipped to the range [0, 1].

The window function suppresses switching near boundaries.

The device retains its state between runs unless reset() or set_state() is used.

*Helper Methods*
set_state(w): sets the next initial state.

reset(): resets the model to w_init = 0.5.

conductance(w): returns conductance for a given state.

resistance(w): returns resistance for a given state.

simulate(...): runs the numerical model.

plot(...): generates standard plots for v(t), i(t), Q-V, and w(t).

*Example Note*
In the included demo, the amplitude is intentionally too small to exceed i_off, so the state does not switch and the I-V curve stays nearly linear.

*File Output*
The plotting helper saves a figure as:

team_qv.png