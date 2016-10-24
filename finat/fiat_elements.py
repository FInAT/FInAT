from __future__ import absolute_import, print_function, division
from six import iteritems

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

    def basis_evaluation(self, order, ps, entity=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''
        fiat_result = self._fiat_element.tabulate(order, ps.points, entity)
        result = {}
        for alpha, table in iteritems(fiat_result):
            # Points be the first dimension, not last.
            table = np.rollaxis(table, -1, 0)

            derivative = sum(alpha)
            if derivative < self._degree:
                point_indices = ps.indices
                point_shape = tuple(index.extent for index in point_indices)
                shape = point_shape + self.index_shape + self.value_shape
                result[alpha] = gem.partial_indexed(
                    gem.Literal(table.reshape(shape)),
                    point_indices
                )
            elif derivative == self._degree:
                # Make sure numerics satisfies theory
                assert np.allclose(table, table.mean(axis=0, keepdims=True))
                result[alpha] = gem.Literal(table[0])
            else:
                # Make sure numerics satisfies theory
                assert np.allclose(table, 0.0)
                result[alpha] = gem.Zero(self.index_shape + self.value_shape)
        return result

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
