import numpy as np
import gem


class UndefinedError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class FiniteElementBase(object):

    def __init__(self):
        pass

    @property
    def cell(self):
        '''The reference cell on which the element is defined.
        '''

        return self._cell

    @property
    def degree(self):
        '''The degree of the embedding polynomial space.

        In the tensor case this is a tuple.
        '''

        return self._degree

    @property
    def entity_dofs(self):
        '''The map of topological entities to degrees of
        freedom for the finite element.

        Note that entity numbering needs to take into account the tensor case.
        '''

        raise NotImplementedError

    @property
    def entity_closure_dofs(self):
        '''The map of topological entities to degrees of
        freedom on the closure of those entities for the finite element.'''

        raise NotImplementedError

    @property
    def index_shape(self):
        '''A tuple indicating the number of degrees of freedom in the
        element. For example a scalar quadratic Lagrange element on a triangle
        would return (6,) while a vector valued version of the same element
        would return (6, 2)'''

        raise NotImplementedError

    @property
    def value_shape(self):
        '''A tuple indicating the shape of the element.'''

        raise NotImplementedError

    def get_indices(self):
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        return tuple(gem.Index(extent=d) for d in self.index_shape)

    def get_value_indices(self):
        '''A tuple of GEM :class:`~gem.Index` of the correct extents to loop over
        the value shape of this element.'''

        return tuple(gem.Index(extent=d) for d in self.value_shape)

    def basis_evaluation(self, q, entity=None, derivative=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param index: the basis function index.
        :param q: the quadrature rule.
        :param q_index: the quadrature index.
        :param entity: the cell entity on which to tabulate.
        :param derivative: the derivative to take of the basis functions.
        '''

        raise NotImplementedError

    @property
    def preferred_quadrature(self):
        '''A list of quadrature rules whose structure this element is capable
        of exploiting. Each entry in the list should be a pair (rule,
        degree), where the degree might be `None` if the element has
        no preferred quadrature degree.'''

        return ()

    def dual_evaluation(self, kernel_data):
        '''Return code for evaluating an expression at the dual set.

        Note: what does the expression need to look like?
        '''

        raise NotImplementedError

    def __hash__(self):
        """Elements are equal if they have the same class, degree, and cell."""

        return hash((type(self), self._cell, self._degree))

    def __eq__(self, other):
        """Elements are equal if they have the same class, degree, and cell."""

        return type(self) == type(other) and self._cell == other._cell and\
            self._degree == other._degree
