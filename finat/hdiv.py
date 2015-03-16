from finiteelementbase import FiatElementBase
from ast import Recipe, IndexSum, LeviCivita
import FIAT
import indices
from derivatives import div, grad, curl


class HDivElement(FiatElementBase):
    def __init__(self, cell, degree):
        super(HDivElement, self).__init__(cell, degree)

    def basis_evaluation(self, q, kernel_data, derivative=None, pullback=True):

        phi = self._tabulated_basis(q.points, kernel_data, derivative)

        i = indices.BasisFunctionIndex(self._fiat_element.space_dimension())

        tIndex = lambda: indices.DimensionIndex(kernel_data.tdim)
        gIndex = lambda: indices.DimensionIndex(kernel_data.gdim)

        alpha = tIndex()

        # The lambda functions here prevent spurious instantiations of invJ and detJ
        J = lambda: kernel_data.J
        invJ = lambda: kernel_data.invJ
        detJ = lambda: kernel_data.detJ

        if derivative is None:
            if pullback:
                beta = alpha
                alpha = gIndex()
                expr = IndexSum((beta,), J()[alpha, beta] * phi[beta, i, q] /
                                detJ())
            else:
                expr = phi[alpha, i, q]
            ind = ((alpha,), (i,), (q,))
        elif derivative is div:
            if pullback:
                expr = IndexSum((alpha,), phi[alpha, alpha, i, q] / detJ())
            else:
                expr = IndexSum((alpha,), phi[alpha, alpha, i, q])
            ind = ((), (i,), (q,))
        elif derivative is grad:
            if pullback:
                beta = tIndex()
                gamma = gIndex()
                delta = gIndex()
                expr = IndexSum((alpha, beta), J()[gamma, alpha] * invJ()[beta, delta] *
                                phi[alpha, beta, i, q]) / detJ()
                ind = ((gamma, delta), (i,), (q,))
            else:
                beta = indices.DimensionIndex(kernel_data.tdim)
                expr = phi[alpha, beta, i, q]
                ind = ((alpha, beta), (i,), (q,))
        elif derivative is curl:
            beta = indices.DimensionIndex(kernel_data.tdim)
            if pullback:
                d = kernel_data.gdim
                gamma = gIndex()
                delta = gIndex()
                zeta = gIndex()
                expr = LeviCivita((zeta,), (gamma, delta),
                                  IndexSum((alpha, beta), J()[gamma, alpha] * invJ()[beta, delta] *
                                           phi[alpha, beta, i, q])) / detJ()
            else:
                d = kernel_data.tdim
                zeta = tIndex()
                expr = LeviCivita((zeta,), (alpha, beta), phi[alpha, beta, i, q])
            if d == 2:
                expr = expr.replace_indices((zeta, 2))
                ind = ((), (i,), (q,))
            elif d == 3:
                ind = ((zeta,), (i,), (q,))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return Recipe(ind, expr)


class RaviartThomas(HDivElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.RaviartThomas(cell, degree)


class BrezziDouglasMarini(HDivElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasMarini(cell, degree)


class BrezziDouglasFortinMarini(HDivElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasFortinMarini(cell, degree)
