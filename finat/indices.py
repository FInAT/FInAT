from . import ast
import pymbolic.primitives as p
from pymbolic.mapper.stringifier import StringifyMapper
import math
from .points import TensorPointSet

__all__ = ["PointIndex", "TensorPointIndex", "BasisFunctionIndex",
           "TensorBasisFunctionIndex",
           "SimpliciallyGradedBasisFunctionIndex",
           "DimensionIndex"]


class IndexBase(ast.Variable):
    '''Base class for symbolic index objects.'''
    def __init__(self, extent, name):
        super(IndexBase, self).__init__(name)
        if isinstance(extent, slice):
            self._extent = extent
        elif (isinstance(extent, int) or
              isinstance(extent, p.Expression)):
            self._extent = slice(extent)
        else:
            raise TypeError("Extent must be a slice or an int")
        self._color = "yellow"

        self.start = self._extent.start
        self.stop = self._extent.stop
        self.step = self._extent.step

    @property
    def extent(self):
        '''A slice indicating the values this index can take.'''
        return self._extent

    @property
    def length(self):
        '''The number of values this index can take.'''
        start = self._extent.start or 0
        stop = self._extent.stop
        step = self._extent.step or 1

        return int(math.ceil((stop - start) / step))

    @property
    def as_range(self):
        """Convert a slice to a range. If the range has expressions as bounds,
        evaluate them.
        """

        return range(int(self._extent.start or 0),
                     int(self._extent.stop),
                     int(self._extent.step or 1))

    @property
    def _str_extent(self):

        return "%s=(%s:%s:%s)" % (str(self),
                                  self._extent.start or 0,
                                  self._extent.stop,
                                  self._extent.step or 1)

    mapper_method = "map_index"

    def get_mapper_method(self, mapper):

        if isinstance(mapper, StringifyMapper):
            return mapper.map_variable
        else:
            raise AttributeError()

    def __repr__(self):

        return "%s(%s)" % (self.__class__.__name__, self.name)

    def set_error(self):
        self._error = True


class PointIndexBase(object):
    # Marker class for point indices.
    pass


class PointIndex(IndexBase, PointIndexBase):
    '''An index running over a set of points, for example quadrature points.'''
    def __init__(self, pointset):

        self.points = pointset

        name = 'q_' + str(PointIndex._count)
        PointIndex._count += 1

        super(PointIndex, self).__init__(pointset.extent, name)

    _count = 0


class TensorIndex(IndexBase):
    """A mixin to create tensor product indices."""
    def __init__(self, factors):

        self.factors = factors

        name = "_x_".join(f.name for f in factors)

        super(TensorIndex, self).__init__(-1, name)


class TensorPointIndex(TensorIndex, PointIndexBase):
    """An index running over a set of points which have a tensor product
    structure. This index is actually composed of multiple factors."""
    def __init__(self, *args):

        if isinstance(args[0], TensorPointSet):
            assert len(args) == 1

            self.points = args[0]

            factors = [PointIndex(f) for f in args[0].factor_sets]
        else:
            factors = args

        super(TensorPointIndex, self).__init__(factors)

    def __getattr__(self, name):

        if name == "_error":
            if any([hasattr(x, "_error") for x in self.factors]):
                return True

        raise AttributeError


class BasisFunctionIndex(IndexBase):
    '''An index over a local set of basis functions.
    E.g. test functions on an element.'''
    def __init__(self, extent):

        name = 'i_' + str(BasisFunctionIndex._count)
        BasisFunctionIndex._count += 1

        super(BasisFunctionIndex, self).__init__(extent, name)

    _count = 0


class TensorBasisFunctionIndex(TensorIndex):
    """An index running over a set of basis functions which have a tensor
    product structure. This index is actually composed of multiple
    factors.
    """
    def __init__(self, *args):

        assert all([isinstance(a, BasisFunctionIndex) for a in args])

        super(TensorBasisFunctionIndex, self).__init__(args)

    def __getattr__(self, name):

        if name == "_error":
            if any([hasattr(x, "_error") for x in self.factors]):
                return True

        raise AttributeError


class SimpliciallyGradedBasisFunctionIndex(BasisFunctionIndex):
    '''An index over simplicial polynomials with a grade, such
    as Dubiner or Bernstein.  Implies a simplicial iteration space.'''
    def __init__(self, sdim, deg):

        # creates name and increments counter
        super(SimpliciallyGradedBasisFunctionIndex, self).__init__(-1)

        self.factors = [BasisFunctionIndex(deg + 1)]

        def mysum(vals):
            return reduce(lambda a, b: a + b, vals, 0)

        for sd in range(1, sdim):
            acur = BasisFunctionIndex(deg + 1 - mysum(self.factors))
            self.factors.append(acur)

    def __getattr__(self, name):

        if name == "_error":
            if any([hasattr(x, "_error") for x in self.factors]):
                return True

        raise AttributeError


class DimensionIndex(IndexBase):
    '''An index over data dimension. For example over topological,
    geometric or vector components.'''
    def __init__(self, extent):

        name = u'\u03B1_'.encode("utf-8") + str(DimensionIndex._count)
        DimensionIndex._count += 1

        super(DimensionIndex, self).__init__(extent, name)

    _count = 0
