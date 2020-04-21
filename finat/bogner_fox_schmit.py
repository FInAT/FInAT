import FIAT
import gem
from gem import ListTensor

from finat.physically_mapped import PhysicallyMappedElement, Citations
from finat.tensor_product import TensorProductElement, factor_point_set
from finat.hermite import Hermite


class CoordinateMapping1D(object):
    def __init__(self, coordinate_mapping, dimension, direction):
        self.cm = coordinate_mapping
        self.dimension = dimension
        self.direction = direction

    def jacobian_at(self, vertex_1d):
        vertex_nd = vertex_1d + ((0.0,) * (self.dimension-1))
        jac_nd = self.cm.jacobian_at(vertex_nd)
        i = self.direction
        return ListTensor([[jac_nd[i, i]]])

    def cell_size(self):
        packing = tuple([0] * (self.dimension-1))
        return gem.partial_indexed(self.cm.cell_size(), packing)


class BognerFoxSchmit(PhysicallyMappedElement, TensorProductElement):
    def __init__(self, cell, degree):
        if degree != 3:
            raise ValueError("Degree must be 3 for Bogner-Fox-Schmit element")

        dim = cell.get_dimension()
        if isinstance(dim, tuple):
            dim = len(dim)
        self.dimension = dim
        TensorProductElement.__init__(self, tuple(Hermite(FIAT.reference_element.UFCInterval(),
                                                          degree)
                                                  for _ in range(dim)))

    def basis_transformation(self, coordinate_mapping):
        raise NotImplementedError

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        entities = self._factor_entity(entity)
        entity_dim, _ = zip(*entities)
        ps_factors = factor_point_set(self.cell, entity_dim, ps)
        factor_results = []
        for (i, (fe, ps_, e)) in enumerate(zip(self.factors, ps_factors, entities)):
            subcoordinatemap = CoordinateMapping1D(coordinate_mapping, self.dimension, i)
            factor_results.append(fe.basis_evaluation(order, ps_, e, coordinate_mapping=subcoordinatemap))
        evaluations = self._merge_evaluations(factor_results)
        return evaluations

    def entity_support_dofs(self):
        raise NotImplementedError