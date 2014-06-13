import pymbolic.primitives as p


class IndexBase(p.Variable):
    '''Base class for symbolic index objects.'''
    def __init__(self, extent, name):
        super(IndexBase, self).__init__(name)
        if isinstance(extent, slice):
            self._extent = extent
        elif isinstance(extent, int):
            self._extent = slice(extent)
        else:
            raise TypeError("Extent must be a slice or an int")

    @property
    def extent(self):
        '''A slice indicating the values this index can take.'''
        return self._extent


class PointIndex(IndexBase):
    '''An index running over a set of points, for example quadrature points.'''
    def __init__(self, extent):

        name = 'q_' + str(PointIndex._count)
        PointIndex._count += 1

        super(PointIndex, self).__init__(extent, name)

    _count = 0


class BasisFunctionIndex(IndexBase):
    '''An index over a local set of basis functions.
    E.g. test functions on an element.'''
    def __init__(self, extent):

        name = 'i_' + str(BasisFunctionIndex._count)
        BasisFunctionIndex._count += 1

        super(BasisFunctionIndex, self).__init__(extent, name)

    _count = 0


class DimensionIndex(IndexBase):
    '''An index over data dimension. For example over topological,
    geometric or vector components.'''
    def __init__(self, extent):

        name = 'alpha_' + str(DimensionIndex._count)
        DimensionIndex._count += 1

        super(DimensionIndex, self).__init__(extent, name)

    _count = 0
