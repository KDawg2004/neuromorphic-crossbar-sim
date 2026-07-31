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

        y_raw = self.crossbar.forward_step(dt)


        if len(y_raw) % 2 != 0:
            raise RuntimeError(
                f"Crossbar column count {len(y_raw)} is not even, cannot pair for differential readout."
            )

        i_plus = y_raw[0::2]
        i_minus = y_raw[1::2]
        y = i_plus - i_minus
        
        return y