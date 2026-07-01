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

print("\n=== Test 4: Conservation check, 4x4 mixed crossbar ===")

cb4 = Crossbar(4, 4, R_row=10.0, R_col=10.0)

for r in range(4):
    for c in range(4):
        if c % 2 == 0:
            cb4.set_device(r, c, TEAMMemristor(
                k_off=1.333, k_on=-1.333, alpha_off=2, alpha_on=2,
                i_off=0.5e-3, i_on=-0.5e-3, G_on=1/500, G_off=1/5000,
            ))
        else:
            cb4.set_device(r, c, BiolekMemcapacitor(
                Cmin=50e-9, Cmax=200e-9, Cinit=100e-9, k=1e7, p=1,
            ))

row_inputs = [0.9, 0.6, 0.3, -0.3]
cb4.apply_row_inputs(row_inputs)

V_nodes = cb4.solve_node_voltages(dt)

I_row_injected = []
for row in range(4):
    n0 = row * 4
    I = (row_inputs[row] - V_nodes[n0]) / cb4.R_row
    I_row_injected.append(I)

I_col_ground = []
for col in range(4):
    n_last = (4 - 1) * 4 + col
    I = V_nodes[n_last] / cb4.R_col
    I_col_ground.append(I)

total_in = sum(I_row_injected)
total_out = sum(I_col_ground)
diff = abs(total_in - total_out)

print(f"  Total current injected (rows) : {total_in:.6e}")
print(f"  Total current to ground (cols): {total_out:.6e}")
print(f"  Conservation diff             : {diff:.2e}")
print(f"  Node voltages:\n{V_nodes.reshape(4,4)}")

print("\n=== Test 5: Per-node KCL residual check ===")

g_row = 1.0 / cb4.R_row
g_col = 1.0 / cb4.R_col
V_grid = V_nodes.reshape(4, 4)

max_residual = 0.0

for row in range(4):
    for col in range(4):
        residual = 0.0

        # Row wire: left
        if col == 0:
            residual += (row_inputs[row] - V_grid[row, col]) * g_row
        else:
            residual += (V_grid[row, col - 1] - V_grid[row, col]) * g_row

        # Row wire: right
        if col < 3:
            residual -= (V_grid[row, col] - V_grid[row, col + 1]) * g_row

        # Column wire: above
        if row > 0:
            residual += (V_grid[row - 1, col] - V_grid[row, col]) * g_col

        # Column wire: below or ground
        if row < 3:
            residual -= (V_grid[row, col] - V_grid[row + 1, col]) * g_col
        else:
            residual -= V_grid[row, col] * g_col

        # Device current
        device = cb4.devices[row][col]
        G = device.current_conductance(dt)
        I_eq = device.current_offset(dt)
        residual -= (G * V_grid[row, col] + I_eq)

        max_residual = max(max_residual, abs(residual))
        print(f"  node ({row},{col}): residual = {residual:.3e}")

print(f"\n  Max residual: {max_residual:.3e}")