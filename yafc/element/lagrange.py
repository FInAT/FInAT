import pymbolic.primitives as p
from finiteelementbase import FiniteElementBase, Recipe
from utils import doc_inherit, IndexSum
import FIAT
import indices

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

    @doc_inherit
    def basis_evaluation(self, points, kernel_data, derivative=None):

        # updates the requisite static data, which in this case
        # is just the matrix.
        static_key = (id(self), id(points), id(derivative))

        static_data = kernel_data.static
        fiat_element = self._fiat_element

        if static_key in static_data:
            phi = static_data[static_key][0]
        else:
            phi = p.Variable('phi_e' + str(self._id))
            # FIXME: for derivative != None, we've got to reengineer this.
            data = fiat_element.tabulate(0, points.points)[
                tuple([0]*points.points.shape[1])]
            static_data[static_key] = (phi, lambda: data)

        # Note for derivative you get a spatial index in here too.
        i = indices.BasisFunctionIndex(fiat_element.space_dimension())
        q = indices.PointIndex(points.points.shape[0])

        ind = [i, q]

        instructions = [phi[i, q]]

        params = []

        return Recipe(ind, instructions, params)

    @doc_inherit
    def field_evaluation(self, field_var, points,
                         kernel_data, derivative=None):

        # updates the requisite static data, which in this case
        # is just the matrix.
        static_key = (id(self), id(points), id(derivative))

        static_data = kernel_data.static
        fiat_element = self._fiat_element

        if static_key in static_data:
            phi = static_data[static_key][0]
        else:
            phi = p.Variable('phi_e' + str(self._id))
            # FIXME: for derivative != None, we've got to reengineer this.
            data = fiat_element.tabulate(0, points.points)[
                tuple([0]*points.points.shape[1])]
            static_data[static_key] = (phi, lambda: data)

        # Note for derivative you get a spatial index in here too.
        i = indices.BasisFunctionIndex(fiat_element.space_dimension())
        q = indices.PointIndex(points.points.shape[0])

        ind = [q]

        instructions = [IndexSum(i, field_var[i] * phi[i, q])]

        params = [field_var]

        return Recipe(ind, instructions, params)
