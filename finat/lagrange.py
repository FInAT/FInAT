import pymbolic.primitives as p
from finiteelementbase import FiatElementBase
from ast import Recipe, IndexSum
import FIAT
import indices
from derivatives import grad


class ScalarElement(FiatElementBase):
    def __init__(self, cell, degree):
        super(ScalarElement, self).__init__(cell, degree)

    def basis_evaluation(self, q, kernel_data, derivative=None, pullback=True):
        '''Produce the variable for the tabulation of the basis
        functions or their derivative. Also return the relevant indices.

        updates the requisite static kernel data, which in this case
        is just the matrix.
        '''
        if derivative not in (None, grad):
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative

        phi = self._tabulated_basis(q.points, kernel_data, derivative)

        i = indices.BasisFunctionIndex(self._fiat_element.space_dimension())

        if derivative is grad:
            alpha = indices.DimensionIndex(kernel_data.tdim)
            if pullback:
                beta = alpha
                alpha = indices.DimensionIndex(kernel_data.gdim)
                invJ = kernel_data.invJ[(beta, alpha)]
                expr = IndexSum((beta,), invJ * phi[(beta, i, q)])
            else:
                expr = phi[(alpha, i, q)]
            ind = ((alpha,), (i,), (q,))
        else:
            ind = ((), (i,), (q,))
            expr = phi[(i, q)]

        return Recipe(indices=ind, expression=expr)

    def field_evaluation(self, field_var, q,
                         kernel_data, derivative=None, pullback=True):
        if derivative not in (None, grad):
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative

        return super(ScalarElement, self).field_evaluation(
            field_var, q, kernel_data, derivative, pullback)

    def moment_evaluation(self, value, weights, q,
                          kernel_data, derivative=None, pullback=True):
        if derivative not in (None, grad):
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative

        return super(ScalarElement, self).moment_evaluation(
            value, weights, q, kernel_data, derivative, pullback)

    def pullback(self, phi, kernel_data, derivative=None):

        if derivative is None:
            return phi
        elif derivative == grad:
            return None  # dot(invJ, grad(phi))
        else:
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative


class Lagrange(ScalarElement):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(cell, degree)

        self._fiat_element = FIAT.Lagrange(cell, degree)


class DiscontinuousLagrange(ScalarElement):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(cell, degree)

        self._fiat_element = FIAT.DiscontinuousLagrange(cell, degree)
