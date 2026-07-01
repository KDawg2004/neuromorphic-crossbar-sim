class CrossbarLayer:
    """
    A neural network layer backed by a crossbar array.
    """

    def __init__(self, crossbar):
        self.crossbar = crossbar

    def forward(self, x, dt):
        """
        Perform one forward pass through the crossbar.

        Parameters
        ----------
        x : array-like
            Input vector.

        dt : float
            Simulation timestep.

        Returns
        -------
        ndarray
            Output current vector.
        """
        if len(x) != self.crossbar.rows:
            raise ValueError(
                f"Expected {self.crossbar.rows} inputs, got {len(x)}."
            )

        self.crossbar.apply_row_inputs(x)

        y = self.crossbar.compute_column_currents(dt)

        self.crossbar.step(dt)

        return y