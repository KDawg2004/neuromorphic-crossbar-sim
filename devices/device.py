from abc import ABC, abstractmethod


class Device(ABC):
    """
    Abstract base class for all devices.
    """

    @abstractmethod
    def network_step(self, v, dt):
        pass

    @abstractmethod
    def network_current(self, v):
        pass

    @abstractmethod
    def program(self, state):
        """
        Program the device to the requested conductance.
        """
        pass

    @abstractmethod
    def state(self):
        pass