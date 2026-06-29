from abc import ABC, abstractmethod


class Device(ABC):

    @abstractmethod
    def network_step(self, v, dt):
        pass

    @abstractmethod
    def network_current(self, v):
        pass

    @abstractmethod
    def state(self):
        pass