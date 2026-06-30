from devices import TEAMMemristor, BiolekMemcapacitor
from sim.crossbar import Crossbar
import numpy as np

cb = Crossbar(2, 2, R_row=10.0, R_col=0.0)
cb.set_device(
    0, 0,
    TEAMMemristor(
        k_off=1.333,
        k_on=-1.333,
        alpha_off=2,
        alpha_on=2,
        i_off=0.5e-3,
        i_on=-0.5e-3,
        G_on=1/500,
        G_off=1/5000
    )
)

cb.set_device(
    0, 1,
    TEAMMemristor(
        k_off=1.333,
        k_on=-1.333,
        alpha_off=2,
        alpha_on=2,
        i_off=0.5e-3,
        i_on=-0.5e-3,
        G_on=1/500,
        G_off=1/5000
    )
)

cb.set_device(
    1, 0,
    BiolekMemcapacitor(
        Cmin=50e-9,
        Cmax=200e-9,
        Cinit=100e-9,
        k=1e7,
        p=10
    )
)

cb.set_device(
    1, 1,
    BiolekMemcapacitor(
        Cmin=50e-9,
        Cmax=200e-9,
        Cinit=100e-9,
        k=1e7,
        p=10
    )
)

dt   = 1e-4
freq = 1.0
t_end = 3.0
t = np.arange(0, t_end, dt)

col_currents_log = []
state_log = []

for t_i in t:
    v0 = 1.0 * np.sin(2 * np.pi * freq * t_i)
    v1 = 1.0 * np.sin(2 * np.pi * freq * t_i + np.pi/2)  # 90 deg offset
    cb.apply_row_inputs([v0, v1])
    col_currents_log.append(cb.compute_column_currents_mna(dt))
    cb.step(dt)
    state_log.append([
        cb.devices[r][c].state()
        for r in range(2)
        for c in range(2)
    ])

col_currents_log = np.array(col_currents_log)
state_log = np.array(state_log)

print("TEAM G:", cb.devices[0][0].current_conductance(dt))
print("Memcap G:", cb.devices[1][0].current_conductance(dt))
V_nodes = cb.solve_node_voltages(dt)
print(f"Node voltages: {V_nodes}")
print(f"Final states: {state_log[-1]}")
print(f"Final column currents: {col_currents_log[-1]}")