import FIAT

from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase


class DiscontinuousElement(FiniteElementBase):
    """Element wrapper that makes a FInAT element discontinuous."""

    def __init__(self, element):
        super(DiscontinuousElement, self).__init__()
        self.element = element

    @property
    def cell(self):
        return self.element.cell

    @property
    def degree(self):
        return self.element.degree

    @cached_property
    def formdegree(self):
        # Always discontinuous!
        return self.element.cell.get_spatial_dimension()

    @cached_property
    def _entity_dofs(self):
        result = {dim: {i: [] for i in entities}
                  for dim, entities in self.cell.get_topology().items()}
        cell_dimension = self.cell.get_dimension()
        result[cell_dimension][0].extend(range(self.space_dimension()))
        return result

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self.element.space_dimension()

    @property
    def index_shape(self):
        return self.element.index_shape

    @property
    def value_shape(self):
        return self.element.value_shape

    @cached_property
    def fiat_equivalent(self):
        return FIAT.DiscontinuousElement(self.element.fiat_equivalent)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        return self.element.basis_evaluation(order, ps, entity, coordinate_mapping=coordinate_mapping)

    def point_evaluation(self, order, refcoords, entity=None):
        return self.element.point_evaluation(order, refcoords, entity)

    def dual_basis(self):
        raise NotImplementedError(f"{self.__class__.__name__} does not have a dual_basis yet!")

    @property
    def mapping(self):
        return self.element.mapping
