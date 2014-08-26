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

        tIndex = lambda: indices.DimensionIndex(kernel_data.tdim)
        gIndex = lambda: indices.DimensionIndex(kernel_data.gdim)

        alpha = tIndex()

        # The lambda functions here prevent spurious instantiations of invJ and detJ
        invJ = lambda: kernel_data.invJ(points)
        detJ = lambda: kernel_data.detJ(points)

        if derivative is None:
            if pullback:
                beta = alpha
                alpha = gIndex()
                expr = IndexSum((beta,), invJ()[beta, alpha] * phi[(beta, i, q)])
            else:
                expr = phi[(alpha, i, q)]
            ind = ((alpha,), (i,), (q,))
        elif derivative is div:
            if pullback:
                beta = tIndex()
                gamma = gIndex()
                expr = IndexSum((gamma,), invJ()[alpha, gamma] * invJ()[beta, gamma]
                                * phi[(alpha, beta, i, q)])
            else:
                expr = IndexSum((alpha,), phi[(alpha, alpha, i, q)])
            ind = ((), (i,), (q,))
        elif derivative is grad:
            if pullback:
                beta = tIndex()
                gamma = gIndex()
                delta = gIndex()
                expr = IndexSum((alpha, beta), invJ()[alpha, gamma] * invJ(beta, delta)
                                * phi[(alpha, beta, i, q)])
                ind = ((gamma, delta), (i,), (q,))
            else:
                beta = tIndex()
                expr = phi[(alpha, beta, i, q)]
                ind = ((alpha, beta), (i,), (q,))
        elif derivative is curl:
            beta = tIndex()
            d = kernel_data.tdim
            zeta = tIndex()
            if pullback:
                if d == 3:
                    gamma = tIndex()
                    expr = IndexSum((gamma,), kernel_data.J(zeta, gamma) *
                                    LeviCivita((gamma,), (alpha, beta), phi[(alpha, beta, i, q)])) \
                        / detJ()
                elif d == 2:
                    expr = LeviCivita((2,), (alpha, beta), phi[(alpha, beta, i, q)]) \
                        / detJ()
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
