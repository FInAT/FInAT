from six import with_metaclass

from abc import ABCMeta, abstractmethod


class PhysicalGeometry(with_metaclass(ABCMeta)):
    @abstractmethod
    def jacobian_at(self, point):
        pass
