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

        # Additional dimension index along the vector dimension.
        alpha = indices.DimensionIndex(points.points.shape[1])

        return Recipe([alpha] + sr.indices, sr.instructions, sr.depends)
