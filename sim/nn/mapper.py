import numpy as np


class CrossbarProgrammer:
    def __init__(self):
        pass  # no stored G_on/G_off

    def map_weights(self, crossbar, weights):
        rows, cols = weights.shape

        if crossbar.rows != rows or crossbar.cols != cols * 2:
            raise ValueError(...)

        w_max = np.abs(weights).max()
        if w_max == 0:
            w_max = 1.0

        for row in range(rows):
            for col in range(cols):
                w_norm = weights[row, col] / w_max

                device_plus = crossbar.get_device(row, col * 2)
                device_minus = crossbar.get_device(row, col * 2 + 1)

                state_plus, state_minus = self._encode_differential(w_norm)

                device_plus.program(state_plus)
                device_minus.program(state_minus)

    def _encode_differential(self, w_norm):
        mag = abs(w_norm)

        if w_norm >= 0:
            return mag, 0.0
        else:
            return 0.0, mag