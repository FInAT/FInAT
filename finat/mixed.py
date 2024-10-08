import numpy

import gem

from finat.finiteelementbase import FiniteElementBase
from finat.enriched import EnrichedElement


def MixedElement(elements):
    """Constructor function for FEniCS-style mixed elements.

    Implements mixed element using :py:class:`EnrichedElement` and
    value shape transformations with :py:class:`MixedSubElement`.
    """
    sizes = [numpy.prod(element.value_shape, dtype=int)
             for element in elements]
    offsets = [int(offset) for offset in numpy.cumsum([0] + sizes)]
    total_size = offsets.pop()
    return EnrichedElement([MixedSubElement(element, total_size, offset)
                            for offset, element in zip(offsets, elements)])


class MixedSubElement(FiniteElementBase):
    """Element wrapper that flattens value shape and places the flattened
    vector in a longer vector of zeros."""

    def __init__(self, element, size, offset):
        assert 0 <= offset <= size
        assert offset + numpy.prod(element.value_shape, dtype=int) <= size

        super().__init__()
        self.element = element
        self.size = size
        self.offset = offset

    @property
    def cell(self):
        return self.element.cell

    @property
    def complex(self):
        return self.element.complex

    @property
    def degree(self):
        return self.element.degree

    @property
    def formdegree(self):
        return self.element.formdegree

    def entity_dofs(self):
        return self.element.entity_dofs()

    def entity_closure_dofs(self):
        return self.element.entity_closure_dofs()

    def entity_support_dofs(self):
        return self.element.entity_support_dofs()

    def space_dimension(self):
        return self.element.space_dimension()

    @property
    def index_shape(self):
        return self.element.index_shape

    @property
    def value_shape(self):
        return (self.size,)

    def _transform(self, v):
        u = [gem.Zero()] * self.size
        for j, zeta in enumerate(numpy.ndindex(self.element.value_shape)):
            u[self.offset + j] = gem.Indexed(v, zeta)
        return u

    def _transform_evaluation(self, core_eval):
        beta = self.get_indices()
        zeta = self.get_value_indices()

        def promote(table):
            v = gem.partial_indexed(table, beta)
            u = gem.ListTensor(self._transform(v))
            return gem.ComponentTensor(gem.Indexed(u, zeta), beta + zeta)

        return {alpha: promote(table)
                for alpha, table in core_eval.items()}

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        core_eval = self.element.basis_evaluation(order, ps, entity, coordinate_mapping=coordinate_mapping)
        return self._transform_evaluation(core_eval)

    def point_evaluation(self, order, refcoords, entity=None):
        core_eval = self.element.point_evaluation(order, refcoords, entity)
        return self._transform_evaluation(core_eval)

    @property
    def mapping(self):
        return self.element.mapping
