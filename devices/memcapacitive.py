from abc import abstractmethod

from .device import Device


class Memcapacitive(Device):

    @abstractmethod
    def step(self, i, dt):
        pass

    @abstractmethod
    def current(self):
        pass

    @abstractmethod
    def equivalent_current(self, v, dt):
        pass