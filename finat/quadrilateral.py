from __future__ import absolute_import, print_function, division

from FIAT.reference_element import FiredrakeQuadrilateral

from finat.finiteelementbase import FiniteElementBase


class QuadrilateralElement(FiniteElementBase):
    """Class for elements on quadrilaterals.  Wraps a tensor product
    element on an interval x interval cell, but appears on a
    quadrilateral cell to the outside world."""

    def __init__(self, element):
        super(QuadrilateralElement, self).__init__()
        self._cell = FiredrakeQuadrilateral()
        self._degree = None  # Who cares? Not used.

        self.product = element

    def basis_evaluation(self, ps, entity=None, derivative=0):
        """Return code for evaluating the element at known points on the
        reference element.

        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        :param derivative: the derivative to take of the basis functions.
        """
        if entity is None:
            entity = (2, 0)

        # Entity is provided in flattened form (d, i)
        # We factor the entity and construct an appropriate
        # entity id for a TensorProductCell: ((d1, d2), i)
        entity_dim, entity_id = entity
        if entity_dim == 2:
            assert entity_id == 0
            product_entity = ((1, 1), 0)
        elif entity_dim == 1:
            facets = [((0, 1), 0),
                      ((0, 1), 1),
                      ((1, 0), 0),
                      ((1, 0), 1)]
            product_entity = facets[entity_id]
        elif entity_dim == 0:
            raise NotImplementedError("Not implemented for 0 dimension entities")
        else:
            raise ValueError("Illegal entity dimension %s" % entity_dim)

        return self.product.basis_evaluation(ps, product_entity, derivative)

    @property
    def index_shape(self):
        return self.product.index_shape

    @property
    def value_shape(self):
        return self.product.value_shape
