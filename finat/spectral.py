import FIAT

import gem

from finat.fiat_elements import ScalarFiatElement
from finat.point_set import GaussLobattoLegendrePointSet, GaussLegendrePointSet


class GaussLobattoLegendre(ScalarFiatElement):
    """1D continuous element with nodes at the Gauss-Lobatto points."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.GaussLobattoLegendre(cell, degree)
        super(GaussLobattoLegendre, self).__init__(fiat_element)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''

        result = super(GaussLobattoLegendre, self).basis_evaluation(order, ps, entity)
        cell_dimension = self.cell.get_dimension()
        if entity is None or entity == (cell_dimension, 0):  # on cell interior
            space_dim = self.space_dimension()
            if isinstance(ps, GaussLobattoLegendrePointSet) and len(ps.points) == space_dim:
                # Bingo: evaluation points match node locations!
                spatial_dim = self.cell.get_spatial_dimension()
                q, = ps.indices
                r, = self.get_indices()
                result[(0,) * spatial_dim] = gem.ComponentTensor(gem.Delta(q, r), (r,))
        return result


class GaussLegendre(ScalarFiatElement):
    """1D discontinuous element with nodes at the Gauss-Legendre points."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.GaussLegendre(cell, degree)
        super(GaussLegendre, self).__init__(fiat_element)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''

        result = super(GaussLegendre, self).basis_evaluation(order, ps, entity)
        cell_dimension = self.cell.get_dimension()
        if entity is None or entity == (cell_dimension, 0):  # on cell interior
            space_dim = self.space_dimension()
            if isinstance(ps, GaussLegendrePointSet) and len(ps.points) == space_dim:
                # Bingo: evaluation points match node locations!
                spatial_dim = self.cell.get_spatial_dimension()
                q, = ps.indices
                r, = self.get_indices()
                result[(0,) * spatial_dim] = gem.ComponentTensor(gem.Delta(q, r), (r,))
        return result


class FDMLagrange(ScalarFiatElement):
    """1D CG element with FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMLagrange(cell, degree)
        super(FDMLagrange, self).__init__(fiat_element)


class FDMHermite(ScalarFiatElement):
    """1D CG element with FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMHermite(cell, degree)
        super(FDMHermite, self).__init__(fiat_element)
