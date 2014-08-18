import pymbolic.primitives as p
from finiteelementbase import FiniteElementBase
from ast import Recipe, IndexSum
from utils import doc_inherit
import FIAT
import indices
from derivatives import div, grad, curl
import numpy as np


class Lagrange(FiniteElementBase):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__()

        self._cell = cell
        self._degree = degree

        self._fiat_element = FIAT.Lagrange(cell, degree)

    @property
    def entity_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom for the finite element.

        Note that entity numbering needs to take into account the tensor case.
        '''

        return self._fiat_element.entity_dofs()

    @property
    def entity_closure_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element.'''

        return self._fiat_element.entity_dofs()

    @property
    def facet_support_dofs(self):
        '''Return the map of facet id to the degrees of freedom for which the
        corresponding basis functions take non-zero values.'''

        return self._fiat_element.entity_support_dofs()

    def _tabulate(self, points, derivative):

        if derivative is None:
            return self._fiat_element.tabulate(0, points.points)[
                tuple([0] * points.points.shape[1])]
        elif derivative is grad:
            tab = self._fiat_element.tabulate(1, points.points)

            ind = np.eye(points.points.shape[1], dtype=int)

            return np.array([tab[tuple(i)] for i in ind])

        else:
            raise ValueError(
                "Lagrange elements do not have a %s operation") % derivative

    def basis_evaluation(self, points, kernel_data, derivative=None, pullback=True):
        '''Produce the variable for the tabulation of the basis
        functions or their derivative. Also return the relevant indices.

        updates the requisite static kernel data, which in this case
        is just the matrix.
        '''
        static_key = (id(self), id(points), id(derivative))

        static_data = kernel_data.static
        fiat_element = self._fiat_element

        if static_key in static_data:
            phi = static_data[static_key][0]
        else:
            phi = p.Variable(('phi_e' if derivative is None else "dphi_e")
                             + str(self._id))
            data = self._tabulate(points, derivative)
            static_data[static_key] = (phi, lambda: data)

        i = indices.BasisFunctionIndex(fiat_element.space_dimension())
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

    @doc_inherit
    def field_evaluation(self, field_var, points,
                         kernel_data, derivative=None, pullback=True):

        basis = self.basis_evaluation(points, kernel_data, derivative, pullback)
        (d, b, p) = basis.indices
        phi = basis.expression

        expr = IndexSum(b, field_var[b[0]] * phi)

        return Recipe((d, (), p), expr)

    @doc_inherit
    def moment_evaluation(self, value, weights, points,
                          kernel_data, derivative=None, pullback=True):

        basis = self.basis_evaluation(points, kernel_data, derivative, pullback)
        (d, b, p) = basis.indices
        phi = basis.expression

        (d_, b_, p_) = value.indices
        psi = value.replace_indices(zip(d_ + p_, d + p)).expression

        w = weights.kernel_variable("w", kernel_data)

        expr = IndexSum(d + p, psi * phi * w[p])

        return Recipe(((), b + b_, ()), expr)

    @doc_inherit
    def pullback(self, phi, kernel_data, derivative=None):

        if derivative is None:
            return phi
        elif derivative == grad:
            return None  # dot(invJ, grad(phi))
        else:
            raise ValueError(
                "Lagrange elements do not have a %s operation") % derivative
