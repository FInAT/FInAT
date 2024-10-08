import FIAT

import gem

from finat.fiat_elements import ScalarFiatElement, Lagrange, DiscontinuousLagrange
from finat.point_set import GaussLobattoLegendrePointSet, GaussLegendrePointSet


class GaussLobattoLegendre(Lagrange):
    """1D continuous element with nodes at the Gauss-Lobatto points."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.GaussLobattoLegendre(cell, degree)
        super(Lagrange, self).__init__(fiat_element)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''

        result = super().basis_evaluation(order, ps, entity)
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


class GaussLegendre(DiscontinuousLagrange):
    """1D discontinuous element with nodes at the Gauss-Legendre points."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.GaussLegendre(cell, degree)
        super(DiscontinuousLagrange, self).__init__(fiat_element)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''

        result = super().basis_evaluation(order, ps, entity)
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
    """DG element with Legendre polynomials."""

    def __init__(self, cell, degree, variant=None):
        fiat_element = FIAT.Legendre(cell, degree, variant=variant)
        super().__init__(fiat_element)


class IntegratedLegendre(ScalarFiatElement):
    """CG element with integrated Legendre polynomials."""

    def __init__(self, cell, degree, variant=None):
        fiat_element = FIAT.IntegratedLegendre(cell, degree, variant=variant)
        super().__init__(fiat_element)


class FDMLagrange(ScalarFiatElement):
    """1D CG element with FDM shape functions and point evaluation BCs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMLagrange(cell, degree)
        super().__init__(fiat_element)


class FDMDiscontinuousLagrange(ScalarFiatElement):
    """1D DG element with derivatives of FDM shape functions with point evaluation Bcs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMDiscontinuousLagrange(cell, degree)
        super().__init__(fiat_element)


class FDMQuadrature(ScalarFiatElement):
    """1D CG element with FDM shape functions and orthogonalized vertex modes."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMQuadrature(cell, degree)
        super().__init__(fiat_element)


class FDMBrokenH1(ScalarFiatElement):
    """1D Broken CG element with FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMBrokenH1(cell, degree)
        super().__init__(fiat_element)


class FDMBrokenL2(ScalarFiatElement):
    """1D DG element with derivatives of FDM shape functions."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMBrokenL2(cell, degree)
        super().__init__(fiat_element)


class FDMHermite(ScalarFiatElement):
    """1D CG element with FDM shape functions, point evaluation BCs and derivative BCs."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.FDMHermite(cell, degree)
        super().__init__(fiat_element)
