from finiteelementbase import FiatElementBase
from ast import Recipe, IndexSum, LeviCivita
import FIAT
import indices
from derivatives import div, grad, curl


class HCurlElement(FiatElementBase):
    def __init__(self, cell, degree):
        super(HCurlElement, self).__init__(cell, degree)

    def basis_evaluation(self, points, kernel_data, derivative=None, pullback=True):

        phi = self._tabulated_basis(points, kernel_data, derivative)

        i = indices.BasisFunctionIndex(self.fiat_element.space_dimension())
        q = indices.PointIndex(points.points.shape[0])
        alpha = indices.DimensionIndex(kernel_data.tdim)

        if derivative is None:
            if pullback:
                beta = alpha
                alpha = indices.DimensionIndex(kernel_data.gdim)
                expr = IndexSum((beta,), kernel_data.invJ(beta, alpha) * phi[(beta, i, q)])
            else:
                expr = phi[(alpha, i, q)]
            ind = ((alpha,), (i,), (q,))
        elif derivative is div:
            if pullback:
                beta = indices.DimensionIndex(kernel_data.tdim)
                gamma = indices.DimensionIndex(kernel_data.gdim)
                expr = IndexSum((gamma,), kernel_data.invJ(alpha, gamma)
                                * kernel_data.invJ(beta, gamma)
                                * phi[(alpha, beta, i, q)])
            else:
                expr = IndexSum((alpha,), phi[(alpha, alpha, i, q)])
            ind = ((), (i,), (q,))
        elif derivative is grad:
            if pullback:
                beta = indices.DimensionIndex(kernel_data.tdim)
                gamma = indices.DimensionIndex(kernel_data.gdim)
                delta = indices.DimensionIndex(kernel_data.gdim)
                expr = IndexSum((alpha, beta), kernel_data.invJ(alpha, gamma)
                                * kernel_data.invJ(beta, delta)
                                * phi[(alpha, beta, i, q)])
                ind = ((gamma, delta), (i,), (q,))
            else:
                beta = indices.DimensionIndex(kernel_data.tdim)
                expr = phi[(alpha, beta, i, q)]
                ind = ((alpha, beta), (i,), (q,))
        elif derivative is curl:
            beta = indices.DimensionIndex(kernel_data.tdim)
            d = kernel_data.tdim
            zeta = indices.DimensionIndex(d)
            if pullback:
                if d == 3:
                    gamma = indices.DimensionIndex(kernel_data.tdim)
                    expr = IndexSum((gamma,), kernel_data.J(zeta, gamma) *
                                    LeviCivita((gamma,), (alpha, beta), phi[(alpha, beta, i, q)])) \
                        / kernel_data.detJ
                elif d == 2:
                    expr = LeviCivita((2,), (alpha, beta), phi[(alpha, beta, i, q)]) \
                        / kernel_data.detJ
            else:
                if d == 3:
                    expr = LeviCivita((zeta,), (alpha, beta), phi[(alpha, beta, i, q)])
                if d == 2:
                    expr = LeviCivita((2,), (alpha, beta), phi[(alpha, beta, i, q)])
            if d == 2:
                ind = ((), (i,), (q,))
            elif d == 3:
                ind = ((zeta,), (i,), (q,))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return Recipe(ind, expr)
