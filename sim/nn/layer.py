class CrossbarLayer:
    """
    A neural network layer backed by a crossbar array.
    """

    def __init__(self, crossbar):
        self.crossbar = crossbar

    def forward(self, x, dt):
        if len(x) != self.crossbar.rows:
            raise ValueError(
                f"Expected {self.crossbar.rows} inputs, got {len(x)}."
            )

        self.crossbar.apply_row_inputs(x)

        raw = self.crossbar.compute_column_currents(dt)

        if len(raw) % 2 != 0:
            raise RuntimeError(
                f"Crossbar column count {len(raw)} is not even, cannot pair for differential readout."
            )

        i_plus = raw[0::2]
        i_minus = raw[1::2]
        y = i_plus - i_minus

        self.crossbar.step(dt)

        return y