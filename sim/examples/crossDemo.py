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

print("\n=== Test 4a: Conservation check, 4x4 mixed crossbar ===")
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

n_cells = cb4.rows * cb4.cols
V_nodes = cb4.solve_node_voltages(dt)

I_row_injected = []
for row in range(4):
    nr0 = row * 4  # row-rail node at col=0
    I = (row_inputs[row] - V_nodes[nr0]) / cb4.R_row
    I_row_injected.append(I)

I_col_ground = []
for col in range(4):
    nc_last = n_cells + (4 - 1) * 4 + col  # column-rail node at bottom row
    I = V_nodes[nc_last] / cb4.R_col
    I_col_ground.append(I)

total_in = sum(I_row_injected)
total_out = sum(I_col_ground)
diff = abs(total_in - total_out)

print(f"  Total current injected (rows) : {total_in:.6e}")
print(f"  Total current to ground (cols): {total_out:.6e}")
print(f"  Conservation diff             : {diff:.2e}")
print(f"  Row-rail voltages:\n{V_nodes[:n_cells].reshape(4,4)}")
print(f"  Col-rail voltages:\n{V_nodes[n_cells:].reshape(4,4)}")

print("\n=== Test 5a: Per-node KCL residual check (doubled-node topology) ===")

g_row = 1.0 / cb4.R_row if cb4.R_row > 0.0 else None
g_col = 1.0 / cb4.R_col if cb4.R_col > 0.0 else None

max_residual = 0.0

for row in range(cb4.rows):
    for col in range(cb4.cols):

        nr = row * cb4.cols + col
        nc = n_cells + row * cb4.cols + col

        v_row = V_nodes[nr]
        v_col = V_nodes[nc]

        device = cb4.devices[row][col]
        G = device.current_conductance(dt)
        I_eq = device.current_offset(dt)
        i_device = G * (v_row - v_col) + I_eq

        residual_r = 0.0
        if g_row is not None:
            if col == 0:
                residual_r += (cb4.row_inputs[row] - v_row) * g_row
            else:
                residual_r += (V_nodes[row * cb4.cols + (col - 1)] - v_row) * g_row
            if col < cb4.cols - 1:
                residual_r -= (v_row - V_nodes[row * cb4.cols + (col + 1)]) * g_row
        else:
            residual_r = cb4.row_inputs[row] - v_row
        residual_r -= i_device

        residual_c = 0.0
        if g_col is not None:
            if row > 0:
                residual_c += (V_nodes[n_cells + (row - 1) * cb4.cols + col] - v_col) * g_col
            if row < cb4.rows - 1:
                residual_c -= (v_col - V_nodes[n_cells + (row + 1) * cb4.cols + col]) * g_col
            else:
                residual_c -= v_col * g_col
        else:
            residual_c = -v_col
        residual_c += i_device

        max_residual = max(max_residual, abs(residual_r), abs(residual_c))
        print(f"  node ({row},{col}): row_residual={residual_r:.3e}, col_residual={residual_c:.3e}")

print(f"\n  Max residual: {max_residual:.3e}")

print("\n=== Task 1: Ideal-wire vs MNA agreement at R -> 0 ===")

cb_check = Crossbar(2, 2, R_row=1e-6, R_col=1e-6)
cb_check.set_device(0, 0, TEAMMemristor(k_off=1.333, k_on=-1.333, alpha_off=2, alpha_on=2,
                                          i_off=0.5e-3, i_on=-0.5e-3, G_on=1/500, G_off=1/5000))
cb_check.set_device(0, 1, TEAMMemristor(k_off=1.333, k_on=-1.333, alpha_off=2, alpha_on=2,
                                          i_off=0.5e-3, i_on=-0.5e-3, G_on=1/500, G_off=1/5000))
cb_check.set_device(1, 0, TEAMMemristor(k_off=1.333, k_on=-1.333, alpha_off=2, alpha_on=2,
                                          i_off=0.5e-3, i_on=-0.5e-3, G_on=1/500, G_off=1/5000))
cb_check.set_device(1, 1, TEAMMemristor(k_off=1.333, k_on=-1.333, alpha_off=2, alpha_on=2,
                                          i_off=0.5e-3, i_on=-0.5e-3, G_on=1/500, G_off=1/5000))

cb_check.apply_row_inputs([0.5, -0.3])

i_ideal = cb_check._compute_column_currents_IDEAL_WIRE()
i_mna = cb_check.compute_column_currents_mna(dt)

print(f"Ideal: {i_ideal}")
print(f"MNA:   {i_mna}")
print(f"Diff:  {np.abs(i_ideal - i_mna)}")