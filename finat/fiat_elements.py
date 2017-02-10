from __future__ import absolute_import, print_function, division
from six import iteritems

from .finiteelementbase import FiniteElementBase
import FIAT
import gem
import numpy as np


class FiatElementBase(FiniteElementBase):
    """Base class for finite elements for which the tabulation is provided
    by FIAT."""
    def __init__(self, fiat_element):
        super(FiatElementBase, self).__init__()
        self._element = fiat_element

    @property
    def cell(self):
        return self._element.get_reference_element()

    @property
    def degree(self):
        # Requires FIAT.CiarletElement
        return self._element.degree()

    def entity_dofs(self):
        return self._element.entity_dofs()

    def entity_closure_dofs(self):
        return self._element.entity_closure_dofs()

    def space_dimension(self):
        return self._element.space_dimension()

    @property
    def index_shape(self):
        return (self._element.space_dimension(),)

    @property
    def value_shape(self):
        return self._element.value_shape()

    def basis_evaluation(self, order, ps, entity=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''
        space_dimension = self._element.space_dimension()
        value_size = np.prod(self._element.value_shape(), dtype=int)
        fiat_result = self._element.tabulate(order, ps.points, entity)
        result = {}
        for alpha, fiat_table in iteritems(fiat_result):
            if isinstance(fiat_table, Exception):
                result[alpha] = gem.Failure(self.index_shape + self.value_shape, fiat_table)
                continue

            derivative = sum(alpha)
            table_roll = fiat_table.reshape(
                space_dimension, value_size, len(ps.points)
            ).transpose(1, 2, 0)

            exprs = []
            for table in table_roll:
                if derivative < self.degree:
                    point_indices = ps.indices
                    point_shape = tuple(index.extent for index in point_indices)
                    exprs.append(gem.partial_indexed(
                        gem.Literal(table.reshape(point_shape + self.index_shape)),
                        point_indices
                    ))
                elif derivative == self.degree:
                    # Make sure numerics satisfies theory
                    assert np.allclose(table, table.mean(axis=0, keepdims=True))
                    exprs.append(gem.Literal(table[0]))
                else:
                    # Make sure numerics satisfies theory
                    assert np.allclose(table, 0.0)
                    exprs.append(gem.Zero(self.index_shape))
            if self.value_shape:
                beta = self.get_indices()
                zeta = self.get_value_indices()
                result[alpha] = gem.ComponentTensor(
                    gem.Indexed(
                        gem.ListTensor(np.array(
                            [gem.Indexed(expr, beta) for expr in exprs]
                        ).reshape(self.value_shape)),
                        zeta),
                    beta + zeta
                )
            else:
                expr, = exprs
                result[alpha] = expr
        return result


class Regge(FiatElementBase):  # naturally tensor valued
    def __init__(self, cell, degree):
        super(Regge, self).__init__(FIAT.Regge(cell, degree))


class ScalarFiatElement(FiatElementBase):
    @property
    def value_shape(self):
        return ()


class Lagrange(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(FIAT.Lagrange(cell, degree))


class GaussLobattoLegendre(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(GaussLobattoLegendre, self).__init__(FIAT.GaussLobattoLegendre(cell, degree))


class DiscontinuousLagrange(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(DiscontinuousLagrange, self).__init__(FIAT.DiscontinuousLagrange(cell, degree))


class GaussLegendre(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(GaussLegendre, self).__init__(FIAT.GaussLegendre(cell, degree))


class VectorFiatElement(FiatElementBase):
    @property
    def value_shape(self):
        return (self.cell.get_spatial_dimension(),)


class RaviartThomas(VectorFiatElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(FIAT.RaviartThomas(cell, degree))


class DiscontinuousRaviartThomas(VectorFiatElement):
    def __init__(self, cell, degree):
        super(DiscontinuousRaviartThomas, self).__init__(FIAT.DiscontinuousRaviartThomas(cell, degree))


class BrezziDouglasMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super(BrezziDouglasMarini, self).__init__(FIAT.BrezziDouglasMarini(cell, degree))


class BrezziDouglasFortinMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super(BrezziDouglasFortinMarini, self).__init__(FIAT.BrezziDouglasFortinMarini(cell, degree))


class Nedelec(VectorFiatElement):
    def __init__(self, cell, degree):
        super(Nedelec, self).__init__(FIAT.Nedelec(cell, degree))


class NedelecSecondKind(VectorFiatElement):
    def __init__(self, cell, degree):
        super(NedelecSecondKind, self).__init__(FIAT.NedelecSecondKind(cell, degree))
