import pymbolic.primitives as p
from finiteelementbase import FiniteElementBase, Recipe
from utils import doc_inherit, IndexSum
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
            tab = fiat_element.tabulate(1, points.points)

            indices = np.eye(points.points.shape[1], dtype=int)

            return np.array([tab[tuple(i)] for i in indices])

        else:
            raise ValueError(
                "Lagrange elements do not have a %s operation") % derivative

    def _tabulation_variable(self, points, kernel_data, derivative):
        # Produce the variable for the tabulation of the basis
        # functions or their derivative. Also return the relevant indices.

        # updates the requisite static data, which in this case
        # is just the matrix.
        static_key = (id(self), id(points), id(derivative))

        static_data = kernel_data.static
        fiat_element = self._fiat_element

        if static_key in static_data:
            phi = static_data[static_key][0]
        else:
            phi = p.Variable('phi_e' if derivative is None else "dphi_e"
                             + str(self._id))
            data = self._tabulate(points, derivative)
            static_data[static_key] = (phi, lambda: data)

        i = indices.BasisFunctionIndex(fiat_element.space_dimension())
        q = indices.PointIndex(points.points.shape[0])

        ind = [i, q]

        if derivative is grad:
            alpha = indices.DimensionIndex(points.points.shape[1])
            ind = [alpha] + ind

        return phi, ind

    def _weights_variable(self, weights, kernel_data):
        # Produce a variable for the quadrature weights.
        static_key = (id(weights), )

        static_data = kernel_data.static

        if static_key in static_data:
            w = static_data[static_key][0]
        else:
            w = p.Variable('w')
            data = weights.points
            static_data[static_key] = (w, lambda: data)

        return w

    @doc_inherit
    def basis_evaluation(self, points, kernel_data, derivative=None):

        phi, ind = self._tabulation_variable(points, kernel_data, derivative)

        instructions = [phi[ind]]

        depends = []

        return Recipe(ind, instructions, depends)

    @doc_inherit
    def field_evaluation(self, field_var, points,
                         kernel_data, derivative=None):

        phi, ind = self._tabulation_variable(points, kernel_data, derivative)

        if derivative is None:
            free_ind = [ind[-1]]
        else:
            free_ind = [ind[0], ind[-1]]

        i = ind[-2]

        instructions = [IndexSum([i], field_var[i] * phi[ind])]

        depends = [field_var]

        return Recipe(free_ind, instructions, depends)

    @doc_inherit
    def moment_evaluation(self, value, weights, points,
                          kernel_data, derivative=None):

        phi, ind = self._tabulation_variable(points, kernel_data, derivative)
        w = self._weights_variable(weights, kernel_data)

        q = ind[-1]
        if derivative is None:
            sum_ind = [q]
        else:
            sum_ind = [ind[0], q]

        i = ind[-2]

        instructions = [IndexSum(sum_ind, value[sum_ind] * w[q] * phi[ind])]

        depends = [value]

        return Recipe([i], instructions, depends)
