from __future__ import absolute_import, print_function, division

from .point_set import restore_shape
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

    def basis_evaluation(self, ps, entity=None, derivative=0):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        :param derivative: the derivative to take of the basis functions.
        '''

        dim = self.cell.get_spatial_dimension()

        i = self.get_indices()
        vi = self.get_value_indices()
        pi = ps.indices
        di = tuple(gem.Index(extent=dim) for i in range(derivative))

        fiat_tab = self._fiat_element.tabulate(derivative, ps.points, entity)

        # Work out the correct transposition between FIAT storage and ours.
        tr = (2, 0, 1) if self.value_shape else (1, 0)

        e = np.eye(dim, dtype=np.int)
        tensor = np.empty((dim,) * derivative, dtype=np.object)
        it = np.nditer(tensor, flags=['multi_index', 'refs_ok'], op_flags=["writeonly"])
        while not it.finished:
            derivative_multi_index = tuple(e[it.multi_index, :].sum(0))
            it[0] = gem.Indexed(gem.Literal(restore_shape(fiat_tab[derivative_multi_index].transpose(tr), ps)),
                                pi + i + vi)
            it.iternext()

        if derivative:
            tensor = gem.Indexed(gem.ListTensor(tensor), di)
        else:
            tensor = tensor[()]

        return gem.ComponentTensor(tensor, i + vi + di)

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


class Regge(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(Regge, self).__init__(cell, degree)

        self._fiat_element = FIAT.Regge(cell, degree)


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


class DiscontinuousRaviartThomas(VectorFiatElement):
    def __init__(self, cell, degree):
        super(DiscontinuousRaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.DiscontinuousRaviartThomas(cell, degree)


class BrezziDouglasMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super(BrezziDouglasMarini, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasMarini(cell, degree)


class BrezziDouglasFortinMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super(BrezziDouglasFortinMarini, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasFortinMarini(cell, degree)


class Nedelec(VectorFiatElement):
    def __init__(self, cell, degree):
        super(Nedelec, self).__init__(cell, degree)

        self._fiat_element = FIAT.Nedelec(cell, degree)


class NedelecSecondKind(VectorFiatElement):
    def __init__(self, cell, degree):
        super(NedelecSecondKind, self).__init__(cell, degree)

        self._fiat_element = FIAT.NedelecSecondKind(cell, degree)
