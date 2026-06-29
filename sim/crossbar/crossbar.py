"""
crossbar.py

Ideal NxM crossbar array for memdevice simulations.

The Crossbar class stores a matrix of device objects and provides
basic infrastructure for applying row inputs and collecting column
currents. Circuit solving and parasitic resistance will be added
in future revisions.
"""

import numpy as np
from devices import TEAMMemristor, BiolekMemcapacitor


class Crossbar:
    """
    Ideal NxM memdevice crossbar.

    Parameters
    ----------
    rows : int
        Number of word lines.

    cols : int
        Number of bit lines.
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

    def compute_column_currents(self):
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
        Returns (rows*cols,) array of node voltages, flattened row-major.
        """
        n_nodes = self.rows * self.cols
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        g_wire = 1.0 / self.R_row if self.R_row > 0.0 else None

        for row in range(self.rows):
            for col in range(self.cols):
                n = row * self.cols + col
                device = self.devices[row][col]

                # Device companion model
                if device is not None:
                    G = device.current_conductance(dt)
                    I_eq = device.current_offset(dt)
                    A[n, n] -= G
                    b[n] -= I_eq

                # Wire resistance terms
                if g_wire is not None:
                    # Left connection
                    if col == 0:
                        # Source node, apply boundary condition
                        A[n, n] -= g_wire
                        b[n] -= g_wire * self.row_inputs[row]
                    else:
                        # Internal left neighbor
                        A[n, n] -= g_wire
                        A[n, n - 1] += g_wire

                    # Right connection
                    if col < self.cols - 1:
                        A[n, n] -= g_wire
                        A[n, n + 1] += g_wire
                else:
                    # Ideal: node voltage equals row input directly
                    A[n, n] = 1.0
                    b[n] = self.row_inputs[row]
        
        # Column resistance terms
        if self.R_col > 0.0:
            g_col = 1.0 / self.R_col

            # Above neighbor (row > 0 only, column tops float)
            if row > 0:
                A[n, n] -= g_col
                A[n, n - self.cols] += g_col

            # Below neighbor or ground
            if row < self.rows - 1:
                A[n, n] -= g_col
                A[n, n + self.cols] += g_col
            else:
                # Last row: column bottom is grounded
                A[n, n] -= g_col

        return np.linalg.solve(A, b)

    def compute_column_currents_mna(self, dt):
        """
        Solve node voltages then sum device currents per column.
        """
        V_nodes = self.solve_node_voltages(dt)
        currents = np.zeros(self.cols)

        for row in range(self.rows):
            for col in range(self.cols):
                device = self.devices[row][col]
                if device is None:
                    continue
                n = row * self.cols + col
                v = V_nodes[n]
                G = device.current_conductance(dt)
                I_eq = device.current_offset(dt)
                currents[col] += G * v + I_eq

        return currents
    
    def step(self, dt):
        """
        Advance all devices by one timestep.
        """
        V_nodes = self.solve_node_voltages(dt)

        for row in range(self.rows):
            for col in range(self.cols):

                device = self.devices[row][col]

                if device is None:
                    continue

                n = row * self.cols + col
                v = V_nodes[n]

                device.network_step(v, dt)