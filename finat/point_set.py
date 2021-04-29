from abc import ABCMeta, abstractproperty
from itertools import chain, product

import numpy

import gem
from gem.utils import cached_property


class AbstractPointSet(metaclass=ABCMeta):
    """A way of specifying a known set of points, perhaps with some
    (tensor) structure.

    Points, when stored, have shape point_set_shape + (point_dimension,)
    where point_set_shape is () for scalar, (N,) for N element vector,
    (N, M) for N x M matrix etc.
    """

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


class PointSingleton(AbstractPointSet):
    """A point set representing a single point.

    These have a `gem.Literal` expression and no indices."""

    def __init__(self, point):
        """Build a PointSingleton from a single point.

        :arg point: A single point of shape (D,) where D is the dimension of
            the point."""
        point = numpy.asarray(point)
        # 1 point ought to be a 1D array - see docstring above and points method
        assert len(point.shape) == 1
        self.point = point

    @property
    def points(self):
        # Make sure we conform to the expected (# of points, point dimension)
        # shape
        return self.point.reshape(1, -1)

    @property
    def indices(self):
        return ()

    @cached_property
    def expression(self):
        return gem.Literal(self.point)


class PointSet(AbstractPointSet):
    """A basic point set with no internal structure representing a vector of
    points."""

    def __init__(self, points):
        """Build a PointSet from a vector of points

        :arg points: A vector of N points of shape (N, D) where D is the
            dimension of each point."""
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

    def almost_equal(self, other, tolerance=1e-12):
        """Approximate numerical equality of point sets"""
        return type(self) == type(other) and \
            self.points.shape == other.points.shape and \
            numpy.allclose(self.points, other.points, rtol=0, atol=tolerance)


class GaussLegendrePointSet(PointSet):
    """Gauss-Legendre quadrature points on the interval.

    This facilitates implementing discontinuous spectral elements.
    """
    def __init__(self, points):
        super(GaussLegendrePointSet, self).__init__(points)
        assert self.points.shape[1] == 1


class GaussLobattoLegendrePointSet(PointSet):
    """Gauss-Lobatto-Legendre quadrature points on the interval.

    This facilitates implementing continuous spectral elements.
    """
    def __init__(self, points):
        super(GaussLobattoLegendrePointSet, self).__init__(points)
        assert self.points.shape[1] == 1


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

    def almost_equal(self, other, tolerance=1e-12):
        """Approximate numerical equality of point sets"""
        return type(self) == type(other) and \
            len(self.factors) == len(other.factors) and \
            all(s.almost_equal(o, tolerance=tolerance)
                for s, o in zip(self.factors, other.factors))
