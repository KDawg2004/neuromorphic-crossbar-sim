from abc import abstractmethod

from .device import Device


class Memristive(Device):

    @abstractmethod
    def step(self, v, dt):
        pass

    @abstractmethod
    def current(self, v):
        pass

    @abstractmethod
    def conductance(self, w):
        pass