import numpy as np


class UndefinedError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


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

    def moment_evaluation(self, value, weights, points, static_data, derivative=None):
        '''Return code for evaluating:

        .. math::

            \int \mathrm{value} : v\, \mathrm{d}x

        where :math:`v` is a test function or the derivative of a test
        function, and : is the inner product operator.

        :param value: an expression. The free indices in value must match those in v.
        :param weights: a point set of quadrature weights.
        :param static_data: the :class:`.KernelData` object corresponding to the current kernel.
        :param derivative: the derivative to take of the test function.
        '''

        raise NotImplementedError

    def dual_evaluation(self, static_data):
        '''Return code for evaluating an expression at the dual set.

        Note: what does the expression need to look like?
        '''

        raise NotImplementedError


class FiatElement(FiniteElementBase):
    """A finite element for which the tabulation is provided by FIAT."""
    def __init__(self, cell, degree):
        super(FiatElement, self).__init__()

        self._cell = cell
        self._degree = degree

    @property
    def entity_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom for the finite element.

        Note that entity numbering needs to take into account the tensor case.
        '''

        return self._fiat_element.entity_dofs()

    @property
    def entity_closure_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element.'''

        return self._fiat_element.entity_dofs()

    @property
    def facet_support_dofs(self):
        '''Return the map of facet id to the degrees of freedom for which the
        corresponding basis functions take non-zero values.'''

        return self._fiat_element.entity_support_dofs()

    def _tabulate(self, points, derivative):

        if derivative:
            tab = self._fiat_element.tabulate(1, points.points)

            ind = np.eye(points.points.shape[1], dtype=int)

            return np.array([tab[tuple(i)] for i in ind])
        else:
            return self._fiat_element.tabulate(0, points.points)[
                tuple([0] * points.points.shape[1])]
