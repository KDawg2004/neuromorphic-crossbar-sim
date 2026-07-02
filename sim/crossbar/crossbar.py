"""
crossbar.py

Ideal NxM crossbar array for memdevice simulations.

The Crossbar class stores a matrix of device objects and provides
basic infrastructure for applying row inputs and collecting column
currents. Circuit solving and parasitic resistance will be added
in future revisions.
"""

import numpy as np


class Crossbar:
    """
    Ideal NxM memdevice crossbar.

    Parameters
    ----------
    rows : int
        Number of word lines.

    cols : int
        Number of bit lines.

    R_row : float
        Row wire resistance in ohms. Default is 0.0 (ideal).

    R_col : float
        Column wire resistance in ohms. Default is 0.0 (ideal).
    """

    def __init__(self, rows, cols, R_row=0.0, R_col=0.0):

        self.rows = rows
        self.cols = cols
        self.R_row = R_row
        self.R_col = R_col
        # Matrix of device objects
        self.devices = [
            [None for _ in range(cols)]
            for _ in range(rows)
        ]

    def set_device(self, row, col, device):
        """
        Place a device object into the crossbar.
        """

        self.devices[row][col] = device

    def get_device(self, row, col):
        """
        Return the device at (row, col).
        """

        return self.devices[row][col]

    def apply_row_inputs(self, inputs):
        """
        Store row excitation values.

        Parameters
        ----------
        inputs : array-like
            One value per row.
        """

        self.row_inputs = np.asarray(inputs)

    def _compute_column_currents_IDEAL_WIRE(self):
        """
        Sum currents from every device in each column.
        """

        currents = np.zeros(self.cols)

        for row in range(self.rows):
            for col in range(self.cols):

                device = self.devices[row][col]

                if device is None:
                    continue

                v = self.row_inputs[row]

                currents[col] += device.network_current(v)

        return currents
    
    def solve_node_voltages(self, dt):
        """
        Build and solve MNA system for node voltages under row and column wire resistance.

        Node layout: 2 * rows * cols unknowns.
        Row-rail node for (row, col):    index = row * cols + col
        Column-rail node for (row, col): index = rows * cols + row * cols + col

        Device sits between the row-rail node and column-rail node at each crosspoint.
        Row-rail is driven by row_inputs at col=0, chained rightward by R_row.
        Column-rail is grounded at the bottom row, chained upward by R_col.
        """
        n_cells = self.rows * self.cols
        n_nodes = 2 * n_cells
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        def row_node(r, c):
            return r * self.cols + c

        def col_node(r, c):
            return n_cells + r * self.cols + c

        g_wire = 1.0 / self.R_row if self.R_row > 0.0 else None
        g_col = 1.0 / self.R_col if self.R_col > 0.0 else None

        for row in range(self.rows):
            for col in range(self.cols):

                nr = row_node(row, col)
                nc = col_node(row, col)
                device = self.devices[row][col]

                # Device companion model, current flows row-rail -> column-rail
                if device is not None:
                    G = device.current_conductance(dt)
                    I_eq = device.current_offset(dt)

                    A[nr, nr] -= G
                    A[nr, nc] += G
                    b[nr] -= I_eq

                    A[nc, nc] -= G
                    A[nc, nr] += G
                    b[nc] += I_eq

                # Row-rail wire chain
                if g_wire is not None:
                    if col == 0:
                        A[nr, nr] -= g_wire
                        b[nr] -= g_wire * self.row_inputs[row]
                    else:
                        A[nr, nr] -= g_wire
                        A[nr, row_node(row, col - 1)] += g_wire

                    if col < self.cols - 1:
                        A[nr, nr] -= g_wire
                        A[nr, row_node(row, col + 1)] += g_wire
                else:
                    # zero row resistance: row-rail node is forced to source voltage
                    A[nr, :] = 0.0
                    A[nr, nr] = 1.0
                    b[nr] = self.row_inputs[row]

                # Column-rail wire chain
                if g_col is not None:
                    if row > 0:
                        A[nc, nc] -= g_col
                        A[nc, col_node(row - 1, col)] += g_col

                    if row < self.rows - 1:
                        A[nc, nc] -= g_col
                        A[nc, col_node(row + 1, col)] += g_col
                    else:
                        # bottom row: column-rail grounded through R_col
                        A[nc, nc] -= g_col
                else:
                    # zero column resistance: column-rail node forced to ground
                    A[nc, :] = 0.0
                    A[nc, nc] = 1.0
                    b[nc] = 0.0

        return np.linalg.solve(A, b)

    def compute_column_currents_mna(self, dt):
        """
        Solve node voltages then sum device currents per column using MNA
        to account for wire resistance.
        dt: timestep (s)
        Returns: (cols,) array of column currents
        """
        V_nodes = self.solve_node_voltages(dt)
        n_cells = self.rows * self.cols
        currents = np.zeros(self.cols)

        for row in range(self.rows):
            for col in range(self.cols):
                device = self.devices[row][col]
                if device is None:
                    continue

                nr = row * self.cols + col
                nc = n_cells + row * self.cols + col

                v_row = V_nodes[nr]
                v_col = V_nodes[nc]
                v_device = v_row - v_col

                G = device.current_conductance(dt)
                I_eq = device.current_offset(dt)

                currents[col] += G * v_device + I_eq

        return currents
    
    def compute_column_currents(self, dt):
        if self.R_row == 0.0 and self.R_col == 0.0:
            return self._compute_column_currents_IDEAL_WIRE()
        else:
            return self.compute_column_currents_mna(dt)
        
    
    def step(self, dt):
        """
        Advance all devices by one timestep.
        dt: timestep (s)
        """
        V_nodes = self.solve_node_voltages(dt)
        n_cells = self.rows * self.cols

        for row in range(self.rows):
            for col in range(self.cols):

                device = self.devices[row][col]

                if device is None:
                    continue

                nr = row * self.cols + col
                nc = n_cells + row * self.cols + col

                v_device = V_nodes[nr] - V_nodes[nc]

                device.network_step(v_device, dt)