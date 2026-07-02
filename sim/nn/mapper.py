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

                g_plus, g_minus = self._encode_differential(
                    w_norm, device_plus.G_on, device_plus.G_off
                )

                device_plus.set_conductance(g_plus)
                device_minus.set_conductance(g_minus)

    def _encode_differential(self, w_norm, G_on, G_off):
        mag = abs(w_norm)
        g_active = G_off + mag * (G_on - G_off)
        if w_norm >= 0:
            return g_active, G_off
        else:
            return G_off, g_active