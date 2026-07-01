import numpy as np


class WeightMapper:
    """
    Maps neural network weights onto crossbar device conductances.
    """

    def __init__(self):
        pass

    def map_weights(self, crossbar, weights):
        """
        Program a crossbar using a weight matrix.

        Parameters
        ----------
        crossbar : Crossbar
            Target crossbar.

        weights : ndarray
            Weight matrix.
        """

        raise NotImplementedError