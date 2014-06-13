import numpy


class UndefinedError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class PointSetBase(object):
    """A way of specifying a known set of points, perhaps with some
    (tensor) structure."""

    def __init__(self):
        pass

    @property
    def points(self):
        """Return a flattened numpy array of points.

        The array has shape (num_points, topological_dim).
        """

        raise NotImplementedError


class PointSet(PointSetBase):
    """A basic pointset with no internal structure."""

    def __init__(self, points):
        self._points = numpy.array(points)

    @property
    def points(self):
        """Return a flattened numpy array of points.

        The array has shape (num_points, topological_dim).
        """

        return self._points


class Recipe(object):
    """AST snippets and data corresponding to some form of finite element evaluation."""
    def __init__(self, indices, instructions, params):
        self._indices = indices
        self._instructions = instructions
        self._params = params

    @property
    def indices(self):
        '''The free indices in this :class:`Recipe`.'''

        return self._indices

    @property
    def instructions(self):
        '''The actual instructions making up this :class:`Recipe`.'''

        return self._instructions

    @property
    def params(self):
        '''The input fields of this :class:`Recipe`.'''

        return self._params


class FiniteElementBase(object):

    def __init__(self):

        self._id = FiniteElementBase._count
        FiniteElementBase._count += 1

    _count = 0

    @property
    def cell(self):
        '''Return the reference cell on which we are defined.
        '''

        return self._cell

    @property
    def degree(self):
        '''Return the degree of the embedding polynomial space.

        In the tensor case this is a tuple.
        '''

        return self._degree

    @property
    def entity_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom for the finite element.

        Note that entity numbering needs to take into account the tensor case.
        '''

        raise NotImplementedError

    @property
    def entity_closure_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element.'''

        raise NotImplementedError

    @property
    def facet_support_dofs(self):
        '''Return the map of facet id to the degrees of freedom for which the
        corresponding basis functions take non-zero values.'''

        raise NotImplementedError

    def field_evaluation(self, points, static_data, derivative=None):
        '''Return code for evaluating a known field at known points on the
        reference element.

        Derivative is an atomic derivative operation.

        Points is some description of the points to evaluate at

        '''

        raise NotImplementedError

    def basis_evaluation(self, points, static_data, derivative=None):
        '''Return code for evaluating a known field at known points on the
        reference element.

        Points is some description of the points to evaluate at

        '''

        raise NotImplementedError

    def pullback(self, derivative):
        '''Return symbolic information about how this element pulls back
        under this derivative.'''

        raise NotImplementedError

    def moment_evaluation(self, weights, points, static_data, derivative=None):
        '''Return code for evaluating f * v * dx where f is an expression and
        v is a test function.
        '''

        raise NotImplementedError

    def dual_evaluation(self, static_data):
        '''Return code for evaluating an expression at the dual set.

        Note: what does the expression need to look like?
        '''

        raise NotImplementedError
