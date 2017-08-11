from __future__ import absolute_import, print_function, division
from six import iteritems

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
                  for dim, entities in iteritems(self.cell.get_topology())}
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

    def basis_evaluation(self, order, ps, entity=None):
        return self.element.basis_evaluation(order, ps, entity)

    def point_evaluation(self, order, refcoords, entity=None):
        return self.element.point_evaluation(order, refcoords, entity)

    @property
    def mapping(self):
        return self.element.mapping
