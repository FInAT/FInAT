from .finiteelementbase import FiniteElementBase
import FIAT
import gem
import numpy as np


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

        assert entity is None

        dim = self.cell.get_spatial_dimension()

        i = self.get_indices()
        vi = self.get_value_indices()
        qi = q.get_indices()
        di = tuple(gem.Index(extent=dim) for i in range(derivative))

        fiat_tab = self._fiat_element.tabulate(derivative, q.points)

        # Work out the correct transposition between FIAT storage and ours.
        tr = (2, 0, 1) if self.value_shape else (1, 0)

        # Convert the FIAT tabulation into a gem tensor. Note that
        # this does not exploit the symmetry of the derivative tensor.
        if derivative:
            e = np.eye(dim, dtype=np.int)
            tensor = np.empty((dim,) * derivative, dtype=np.object)
            it = np.nditer(tensor, flags=['multi_index', 'refs_ok'], op_flags=["writeonly"])
            while not it.finished:
                derivative_multi_index = tuple(e[it.multi_index, :].sum(0))
                it[0] = gem.Literal(fiat_tab[derivative_multi_index].transpose(tr))
                it.iternext()
            tensor = gem.ListTensor(tensor)
        else:
            tensor = gem.Literal(fiat_tab[(0,) * dim].transpose(tr))

        return gem.ComponentTensor(gem.Indexed(tensor,
                                               di + qi + i + vi),
                                   qi + i + vi + di)

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


class ScalarFiatElement(FiatElementBase):
    @property
    def value_shape(self):
        return ()


class Lagrange(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(cell, degree)

        self._fiat_element = FIAT.Lagrange(cell, degree)


class GaussLobatto(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(GaussLobatto, self).__init__(cell, degree)

        self._fiat_element = FIAT.GaussLobatto(cell, degree)


class DiscontinuousLagrange(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(DiscontinuousLagrange, self).__init__(cell, degree)

        self._fiat_element = FIAT.DiscontinuousLagrange(cell, degree)


class VectorFiatElement(FiatElementBase):
    @property
    def value_shape(self):
        return (self.cell.get_spatial_dimension(),)


class RaviartThomas(VectorFiatElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.RaviartThomas(cell, degree)


class BrezziDouglasMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super(BrezziDouglasMarini, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasMarini(cell, degree)


class BrezziDouglasFortinMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super(BrezziDouglasFortinMarini, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasFortinMarini(cell, degree)
