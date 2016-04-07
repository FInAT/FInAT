from .finiteelementbase import FiatElementBase
import FIAT


class FiatElement(FiatElementBase):
    def __init__(self, cell, degree):
        super(FiatElement, self).__init__(cell, degree)

    # def basis_evaluation(self, q, kernel_data, derivative=None, pullback=True):
    #     '''Produce the variable for the tabulation of the basis
    #     functions or their derivative. Also return the relevant indices.

    #     updates the requisite static kernel data, which in this case
    #     is just the matrix.
    #     '''
    #     if derivative not in (None, grad):
    #         raise ValueError(
    #             "Scalar elements do not have a %s operation") % derivative

    #     phi = self._tabulated_basis(q.points, kernel_data, derivative)

    #     i = indices.BasisFunctionIndex(self._fiat_element.space_dimension())

    #     if derivative is grad:
    #         alpha = indices.DimensionIndex(self.cell.get_spatial_dimension())
    #         if pullback:
    #             beta = alpha
    #             alpha = indices.DimensionIndex(kernel_data.gdim)
    #             invJ = kernel_data.invJ[(beta, alpha)]
    #             expr = IndexSum((beta,), invJ * phi[(beta, i, q)])
    #         else:
    #             expr = phi[(alpha, i, q)]
    #         ind = ((alpha,), (i,), (q,))
    #     else:
    #         ind = ((), (i,), (q,))
    #         expr = phi[(i, q)]

    #     return Recipe(indices=ind, body=expr)


class Lagrange(FiatElement):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(cell, degree)

        self._fiat_element = FIAT.Lagrange(cell, degree)


class GaussLobatto(FiatElement):
    def __init__(self, cell, degree):
        super(GaussLobatto, self).__init__(cell, degree)

        self._fiat_element = FIAT.GaussLobatto(cell, degree)

    # def basis_evaluation(self, q, kernel_data, derivative=None, pullback=True):
    #     '''Produce the variable for the tabulation of the basis
    #     functions or their derivative. Also return the relevant indices.

    #     For basis evaluation with no gradient on a matching
    #     Gauss-Lobatto quadrature, this implements the standard
    #     spectral element diagonal mass trick by returning a delta
    #     function.
    #     '''
    #     if (derivative is None and isinstance(q.points, GaussLobattoPointSet) and
    #             q.length == self._fiat_element.space_dimension()):

    #         i = indices.BasisFunctionIndex(self._fiat_element.space_dimension())

    #         return Recipe(((), (i,), (q,)), Delta((i, q), 1.0))

    #     else:
    #         # Fall through to the default recipe.
    #         return super(GaussLobatto, self).basis_evaluation(q, kernel_data,
    #                                                           derivative, pullback)


class DiscontinuousLagrange(FiatElement):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(cell, degree)

        self._fiat_element = FIAT.DiscontinuousLagrange(cell, degree)
