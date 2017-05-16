from __future__ import absolute_import, print_function, division
from six import iteritems

from FIAT.reference_element import FiredrakeQuadrilateral

from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase


class QuadrilateralElement(FiniteElementBase):
    """Class for elements on quadrilaterals.  Wraps a tensor product
    element on an interval x interval cell, but appears on a
    quadrilateral cell to the outside world."""

    def __init__(self, element):
        super(QuadrilateralElement, self).__init__()
        self.product = element

    @cached_property
    def cell(self):
        return FiredrakeQuadrilateral()

    @property
    def degree(self):
        unique_degree, = set(self.product.degree)
        return unique_degree

    @cached_property
    def _entity_dofs(self):
        entity_dofs = self.product.entity_dofs()
        flat_entity_dofs = {}
        flat_entity_dofs[0] = entity_dofs[(0, 0)]
        flat_entity_dofs[1] = dict(enumerate(
            [v for k, v in sorted(iteritems(entity_dofs[(0, 1)]))] +
            [v for k, v in sorted(iteritems(entity_dofs[(1, 0)]))]
        ))
        flat_entity_dofs[2] = entity_dofs[(1, 1)]
        return flat_entity_dofs

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self.product.space_dimension()

    def basis_evaluation(self, order, ps, entity=None):
        """Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        """
        return self.product.basis_evaluation(order, ps, productise(entity))

    def point_evaluation(self, order, point, entity=None):
        return self.product.point_evaluation(order, point, productise(entity))

    @property
    def index_shape(self):
        return self.product.index_shape

    @property
    def value_shape(self):
        return self.product.value_shape


def productise(entity):
    if entity is None:
        entity = (2, 0)

    # Entity is provided in flattened form (d, i)
    # We factor the entity and construct an appropriate
    # entity id for a TensorProductCell: ((d1, d2), i)
    entity_dim, entity_id = entity
    if entity_dim == 2:
        assert entity_id == 0
        return ((1, 1), 0)
    elif entity_dim == 1:
        facets = [((0, 1), 0),
                  ((0, 1), 1),
                  ((1, 0), 0),
                  ((1, 0), 1)]
        return facets[entity_id]
    elif entity_dim == 0:
        raise NotImplementedError("Not implemented for 0 dimension entities")
    else:
        raise ValueError("Illegal entity dimension %s" % entity_dim)
