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
    def complex(self):
        return self.element.complex

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

    @cached_property
    def entity_permutations(self):
        # Return entity_permutations of the base finite element if it only
        # has cell degrees of freedom; otherwise entity_permutations is not
        # yet implemented for DiscontinuousElement.
        if self.element.entity_dofs() == self.element.entity_closure_dofs():
            return self.element.entity_permutations
        else:
            raise NotImplementedError(f"entity_permutations not yet implemented for a general {type(self)}")

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

    @property
    def dual_basis(self):
        return self.element.dual_basis

    @property
    def mapping(self):
        return self.element.mapping
