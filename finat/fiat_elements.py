from __future__ import absolute_import, print_function, division

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
        di = tuple(gem.Index(extent=dim) for i in range(derivative))

        if derivative < self._degree:
            points = ps.points
            point_indices = ps.indices
        elif derivative == self._degree:
            # Tabulate on cell centre
            points = np.mean(self._cell.get_vertices(), axis=0, keepdims=True)
            entity = (self._cell.get_dimension(), 0)
            point_indices = ()  # no point indices used
        else:
            return gem.Zero(tuple(index.extent for index in i + vi + di))

        fiat_tab = self._fiat_element.tabulate(derivative, points, entity)

        # Work out the correct transposition between FIAT storage and ours.
        tr = (2, 0, 1) if self.value_shape else (1, 0)

        def restore_point_shape(array):
            shape = tuple(index.extent for index in point_indices)
            return array.reshape(shape + array.shape[1:])

        e = np.eye(dim, dtype=np.int)
        tensor = np.empty((dim,) * derivative, dtype=np.object)
        for multi_index in np.ndindex(tensor.shape):
            derivative_multi_index = tuple(e[multi_index, :].sum(axis=0))
            transposed_table = fiat_tab[derivative_multi_index].transpose(tr)
            tensor[multi_index] = gem.Indexed(gem.Literal(restore_point_shape(transposed_table)),
                                              point_indices + i + vi)

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
