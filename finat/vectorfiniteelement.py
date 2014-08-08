from finiteelementbase import FiniteElementBase
from derivatives import div, grad, curl
from utils import doc_inherit, IndexSum
from ast import Recipe
import indices


class VectorFiniteElement(FiniteElementBase):
    def __init__(self, element, dimension):
        super(VectorFiniteElement, self).__init__()

        self._cell = element._cell
        self._degree = element._degree

        self._dimension = dimension

        self._base_element = element

    @doc_inherit
    def basis_evaluation(self, points, kernel_data, derivative=None):

        # Produce the base scalar recipe
        sr = self._base_element.basis_evaluation(points, kernel_data, derivative)

        # Additional dimension index along the vector dimension. Note
        # to self: is this the right order or does this index come
        # after any derivative index?
        alpha = indices.DimensionIndex(points.points.shape[1])

        return Recipe([alpha] + sr.indices, sr.instructions, sr.depends)

    @doc_inherit
    def field_evaluation(self, field_var, points,
                         kernel_data, derivative=None):

        basis = self._base_element.basis_evaluation(self, points,
                                                    kernel_data, derivative)

        alpha = indices.DimensionIndex(points.points.shape[1])
        ind = basis.indices

        if derivative is None:
            free_ind = [alpha, ind[-1]]
        else:
            free_ind = [alpha, ind[0], ind[-1]]

        i = ind[-2]

        instructions = [IndexSum(i, field_var[i, alpha] * basis[ind])]

        depends = [field_var]

        return Recipe(free_ind, instructions, depends)

    @doc_inherit
    def moment_evaluation(self, value, weights, points,
                          kernel_data, derivative=None):

        basis = self._base_element.basis_evaluation(self, points,
                                                    kernel_data, derivative)
        w = weights.kernel_variable("w", kernel_data)
        ind = basis.indices

        q = ind[-1]
        alpha = indices.DimensionIndex(points.points.shape[1])

        if derivative is None:
            sum_ind = [q]
        elif derivative == grad:
            sum_ind = [ind[0], q]
        else:
            raise NotImplementedError()

        value_ind = [alpha] + sum_ind

        instructions = [IndexSum(sum_ind, value[value_ind] * w[q] * basis[ind])]

        depends = [value]

        free_ind = [alpha, ind[-2]]

        return Recipe(free_ind, instructions, depends)

    @doc_inherit
    def pullback(self, phi, kernel_data, derivative=None):

        if derivative is None:
            return phi
        elif derivative == grad:
            return None  # IndexSum(alpha, Jinv[:, alpha] * grad(phi)[:,alpha])
        else:
            raise ValueError(
                "Lagrange elements do not have a %s operation") % derivative
