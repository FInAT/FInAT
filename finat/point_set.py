from __future__ import absolute_import, print_function, division
from six import with_metaclass
from six.moves import range

from abc import ABCMeta, abstractproperty
from itertools import chain, product

import numpy

import gem
from gem.utils import cached_property


class AbstractPointSet(with_metaclass(ABCMeta)):
    """A way of specifying a known set of points, perhaps with some
    (tensor) structure."""

    @abstractproperty
    def points(self):
        """A flattened numpy array of points with shape
        (# of points, point dimension)."""

    @property
    def dimension(self):
        """Point dimension."""
        _, dim = self.points.shape
        return dim

    @abstractproperty
    def indices(self):
        """GEM indices with matching shape and extent to the structure of the
        point set."""

    @abstractproperty
    def expression(self):
        """GEM expression describing the points, with free indices
        ``self.indices`` and shape (point dimension,)."""


class PointSet(AbstractPointSet):
    """A basic point set with no internal structure."""

    def __init__(self, points):
        points = numpy.asarray(points)
        assert len(points.shape) == 2
        self.points = points

    @cached_property
    def points(self):
        pass  # set at initialisation

    @cached_property
    def indices(self):
        return (gem.Index(extent=len(self.points)),)

    @cached_property
    def expression(self):
        return gem.partial_indexed(gem.Literal(self.points), self.indices)


class TensorPointSet(AbstractPointSet):

    def __init__(self, factors):
        self.factors = tuple(factors)

    @cached_property
    def points(self):
        return numpy.array([list(chain(*pt_tuple))
                            for pt_tuple in product(*[ps.points
                                                      for ps in self.factors])])

    @cached_property
    def indices(self):
        return tuple(chain(*[ps.indices for ps in self.factors]))

    @cached_property
    def expression(self):
        result = []
        for point_set in self.factors:
            for i in range(point_set.dimension):
                result.append(gem.Indexed(point_set.expression, (i,)))
        return gem.ListTensor(result)
