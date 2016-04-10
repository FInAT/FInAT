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

    def get_indices():
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        return tuple(gem.Index(d) for d in self.index_shape)

    def get_value_indices():
        '''A tuple of GEM :class:`~gem.Index` of the correct extents to loop over
        the value shape of this element.'''

        return tuple(gem.Index(d) for d in self.value_shape)

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


class FiatElementBase(FiniteElementBase):
    """Base class for finite elements for which the tabulation is provided
    by FIAT."""
    def __init__(self, cell, degree):
        super(FiatElementBase, self).__init__()

        self._cell = cell
        self._degree = degree

    @property
    def index_shape(self):
        return (self._fiat_element.space_dimension(),)

    @property
    def value_shape(self):
        return self._fiat_element.value_shape()

    def basis_evaluation(self, q, entity=None, derivative=0):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param q: the quadrature rule.
        :param entity: the cell entity on which to tabulate.
        :param derivative: the derivative to take of the basis functions.
        '''

        assert entity == None

        dim = self.cell.get_spatial_dimension()

        i = self.get_indices()
        vi = self.get_value_indices()
        qi = q.get_indices()
        di = tuple(gem.Index() for i in range(dim)) 

        fiat_tab = self._fiat_element.tabulate(derivative, q.points)

        # Work out the correct transposition between FIAT storage and ours.
        tr = (2, 0, 1) if self.value_shape else (1, 0)

        # Convert the FIAT tabulation into a gem tensor. Note that
        # this does not exploit the symmetry of the derivative tensor.
        def tabtensor(index = (0,) * dim):
            if sum(index) < derivative:
                return gem.ListTensor([tabtensor(tuple(index[id] + (1 if id == i else 0) for id in range(dim)))
                                       for i in range(dim)])
            else:
                return gem.Literal(fiat_tab[index].transpose(tr))

        return ComponentTensor(Indexed(tabtensor(), di + qi + i + vi), qi + i + vi + di)

    @property
    def entity_dofs(self):
        '''The map of topological entities to degrees of
        freedom for the finite element.

        Note that entity numbering needs to take into account the tensor case.
        '''

        return self._fiat_element.entity_dofs()

    @property
    def entity_closure_dofs(self):
        '''The map of topological entities to degrees of
        freedom on the closure of those entities for the finite element.'''

        return self._fiat_element.entity_closure_dofs()

    @property
    def facet_support_dofs(self):
        '''The map of facet id to the degrees of freedom for which the
        corresponding basis functions take non-zero values.'''

        return self._fiat_element.entity_support_dofs()
