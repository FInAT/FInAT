import pymbolic.primitives as p
from finiteelementbase import FiatElement
from ast import Recipe, IndexSum
import FIAT
import indices
from derivatives import grad


class ScalarElement(FiatElement):
    def __init__(self, cell, degree):
        super(ScalarElement, self).__init__(cell, degree)

    def basis_evaluation(self, points, kernel_data, derivative=None, pullback=True):
        '''Produce the variable for the tabulation of the basis
        functions or their derivative. Also return the relevant indices.

        updates the requisite static kernel data, which in this case
        is just the matrix.
        '''
        if derivative not in (None, grad):
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative

        static_key = (id(self), id(points), id(derivative))

        if static_key in kernel_data.static:
            phi = kernel_data.static[static_key][0]
        else:
            phi = p.Variable((u'\u03C6_e'.encode("utf-8") if derivative is None
                             else u"d\u03C6_e".encode("utf-8")) + str(self._id))
            data = self._tabulate(points, derivative)
            kernel_data.static[static_key] = (phi, lambda: data)

        i = indices.BasisFunctionIndex(self._fiat_element.space_dimension())
        q = indices.PointIndex(points.points.shape[0])

        if derivative is grad:
            alpha = indices.DimensionIndex(points.points.shape[1])
            ind = ((alpha,), (i,), (q,))
            if pullback:
                beta = indices.DimensionIndex(points.points.shape[1])
                invJ = kernel_data.invJ[(beta, alpha)]
                expr = IndexSum((beta,), invJ * phi[(beta, i, q)])
            else:
                expr = phi[(alpha, i, q)]
        else:
            ind = ((), (i,), (q,))
            expr = phi[(i, q)]

        return Recipe(indices=ind, expression=expr)

    def field_evaluation(self, field_var, points,
                         kernel_data, derivative=None, pullback=True):
        if derivative not in (None, grad):
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative

        return super(ScalarElement, self).field_evaluation(
            field_var, points, kernel_data, derivative=None, pullback=True)

    def moment_evaluation(self, value, weights, points,
                          kernel_data, derivative=None, pullback=True):
        if derivative not in (None, grad):
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative

        return super(ScalarElement, self).moment_evaluation(
            value, weights, points, kernel_data, derivative=None, pullback=True)

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
