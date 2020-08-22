from __future__ import absolute_import, print_function, division

from FIAT.reference_element import UFCHexahedron, UFCQuadrilateral
from FIAT.reference_element import compute_unflattening_map, flatten_entities

from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase


class FlattenedDimensions(FiniteElementBase):
    """Class for elements on quadrilaterals and hexahedra.  Wraps a tensor
    product element on a tensor product cell, and flattens its entity
    dimensions."""

    def __init__(self, element):
        super(FlattenedDimensions, self).__init__()
        self.product = element
        self._unflatten = compute_unflattening_map(element.cell.get_topology())

    @cached_property
    def cell(self):
        dim = self.product.cell.get_spatial_dimension()
        if dim == 2:
            return UFCQuadrilateral()
        elif dim == 3:
            return UFCHexahedron()
        else:
            raise NotImplementedError("Cannot guess cell for spatial dimension %s" % dim)

    @property
    def degree(self):
        unique_degree, = set(self.product.degree)
        return unique_degree

    @property
    def formdegree(self):
        return self.product.formdegree

    @cached_property
    def _entity_dofs(self):
        return flatten_entities(self.product.entity_dofs())

    @cached_property
    def _entity_support_dofs(self):
        return flatten_entities(self.product.entity_support_dofs())

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self.product.space_dimension()

    @cached_property
    def fiat_equivalent(self):
        # On-demand loading of FIAT module
        from FIAT.tensor_product import FlattenedDimensions

        return FlattenedDimensions(self.product.fiat_equivalent)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        """Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        """
        if entity is None:
            entity = (self.cell.get_spatial_dimension(), 0)

        return self.product.basis_evaluation(order, ps, self._unflatten[entity])

    def point_evaluation(self, order, point, entity=None):
        if entity is None:
            entity = (self.cell.get_spatial_dimension(), 0)

        return self.product.point_evaluation(order, point, self._unflatten[entity])

    @property
    def index_shape(self):
        return self.product.index_shape

    @property
    def value_shape(self):
        return self.product.value_shape

    @property
    def mapping(self):
        return self.product.mapping
