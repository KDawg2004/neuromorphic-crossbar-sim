import numpy as np
from devices import TEAMMemristor
from sim.crossbar import Crossbar

dt = 1e-4
freq = 1.0
amp = 0.9
t_end = 3.0
t = np.arange(0, t_end, dt)

def make_team():
    return TEAMMemristor(
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

# --- Test 1: R_row=0 MNA matches ideal ---
print("=== Test 1: MNA (R_row=0) vs ideal ===")
cb_ideal  = Crossbar(2, 2, R_row=0.0)
cb_mna    = Crossbar(2, 2, R_row=0.0)

for r in range(2):
    for c in range(2):
        cb_ideal.set_device(r, c, make_team())
        cb_mna.set_device(r, c, make_team())

for t_i in t:
    v = amp * np.sin(2 * np.pi * freq * t_i)
    cb_ideal.apply_row_inputs([v, v])
    cb_mna.apply_row_inputs([v, v])
    cb_ideal.step(dt)
    cb_mna.step(dt)

v = amp * np.sin(2 * np.pi * freq * t[-1])
cb_ideal.apply_row_inputs([v, v])
cb_mna.apply_row_inputs([v, v])

ideal_currents = cb_ideal.compute_column_currents()
mna_currents   = cb_mna.compute_column_currents_mna(dt)

for col in range(2):
    diff = abs(ideal_currents[col] - mna_currents[col])
    print(f"  col {col}: ideal={ideal_currents[col]:.6e}  mna={mna_currents[col]:.6e}  diff={diff:.2e}")

# --- Test 2: R_row > 0 produces lower device voltages ---
print("\n=== Test 2: R_row=10 reduces effective voltage ===")
cb_r = Crossbar(2, 2, R_row=10.0)
for r in range(2):
    for c in range(2):
        cb_r.set_device(r, c, make_team())

v = 0.9
cb_r.apply_row_inputs([v, v])
V_nodes = cb_r.solve_node_voltages(dt)
for row in range(2):
    for col in range(2):
        n = row * 2 + col
        print(f"  node ({row},{col}): V={V_nodes[n]:.6f}  (applied={v})")

print("\n=== Test 3: R_col=10 reduces voltage further ===")
cb_rc = Crossbar(2, 2, R_row=10.0, R_col=10.0)
for r in range(2):
    for c in range(2):
        cb_rc.set_device(r, c, make_team())

v = 0.9
cb_rc.apply_row_inputs([v, v])
V_nodes = cb_rc.solve_node_voltages(dt)
for row in range(2):
    for col in range(2):
        n = row * 2 + col
        print(f"  node ({row},{col}): V={V_nodes[n]:.6f}  (applied={v})")