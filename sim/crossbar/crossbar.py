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

    def __init__(self, rows, cols):

        self.rows = rows
        self.cols = cols

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
    
    def step(self, dt):
        """
        Advance all devices by one timestep.
        """

        for row in range(self.rows):
            for col in range(self.cols):

                device = self.devices[row][col]

                if device is None:
                    continue

                v = self.row_inputs[row]

                device.network_step(v, dt)