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


class Legendre(ScalarFiatElement):
    """1D DG element with Legendre polynomials."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.Legendre(cell, degree)
        super(Legendre, self).__init__(fiat_element)


class IntegratedLegendre(ScalarFiatElement):
    """1D CG element with integrated Legendre polynomials."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.IntegratedLegendre(cell, degree)
        super(IntegratedLegendre, self).__init__(fiat_element)


class FDMLagrange(ScalarFiatElement):
    """1D CG element with FDM shape functions and point evaluation BCs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMLagrange(cell, degree)
        super(FDMLagrange, self).__init__(fiat_element)


class FDMDiscontinuousLagrange(ScalarFiatElement):
    """1D DG element with derivatives of FDM shape functions with point evaluation Bcs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMDiscontinuousLagrange(cell, degree)
        super(FDMDiscontinuousLagrange, self).__init__(fiat_element)


class FDMQuadrature(ScalarFiatElement):
    """1D CG element with FDM shape functions and orthogonalized vertex modes."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMQuadrature(cell, degree)
        super(FDMQuadrature, self).__init__(fiat_element)


class FDMBrokenH1(ScalarFiatElement):
    """1D Broken CG element with FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMBrokenH1(cell, degree)
        super(FDMBrokenH1, self).__init__(fiat_element)


class FDMBrokenL2(ScalarFiatElement):
    """1D DG element with derivatives of FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMBrokenL2(cell, degree)
        super(FDMBrokenL2, self).__init__(fiat_element)


class FDMHermite(ScalarFiatElement):
    """1D CG element with FDM shape functions, point evaluation BCs and derivative BCs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMHermite(cell, degree)
        super(FDMHermite, self).__init__(fiat_element)
