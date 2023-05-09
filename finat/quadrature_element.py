from finat.point_set import UnknownPointSet
from functools import reduce

import numpy

import FIAT

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase, MappingStr
from finat.quadrature import make_quadrature, AbstractQuadratureRule


def make_quadrature_element(fiat_ref_cell, degree, scheme="default"):
    """Construct a :class:`QuadratureElement` from a given a reference
    element, degree and scheme.

    :param fiat_ref_cell: The FIAT reference cell to build the
        :class:`QuadratureElement` on.
    :param degree: The degree of polynomial that the rule should
        integrate exactly.
    :param scheme: The quadrature scheme to use - e.g. "default",
        "canonical" or "KMV".
    :returns: The appropriate :class:`QuadratureElement`
    """
    rule = make_quadrature(fiat_ref_cell, degree, scheme)
    return QuadratureElement(fiat_ref_cell, rule)


class QuadratureElement(FiniteElementBase):
    """A set of quadrature points pretending to be a finite element."""

    def __init__(self, fiat_ref_cell, rule):
        """Construct a :class:`QuadratureElement`.

        :param fiat_ref_cell: The FIAT reference cell to build the
            :class:`QuadratureElement` on
        :param rule: A :class:`AbstractQuadratureRule` to use
        """
        self.cell = fiat_ref_cell
        if not isinstance(rule, AbstractQuadratureRule):
            raise TypeError("rule is not an AbstractQuadratureRule")
        if fiat_ref_cell.get_spatial_dimension() != rule.point_set.dimension:
            raise ValueError("Cell dimension does not match rule's point set dimension")
        self._rule = rule

    @cached_property
    def cell(self):
        pass  # set at initialisation

    @property
    def degree(self):
        raise NotImplementedError("QuadratureElement does not represent a polynomial space.")

    @property
    def formdegree(self):
        return None

    @cached_property
    def _entity_dofs(self):
        # Inspired by ffc/quadratureelement.py
        entity_dofs = {dim: {entity: [] for entity in entities}
                       for dim, entities in self.cell.get_topology().items()}
        entity_dofs[self.cell.get_dimension()] = {0: list(range(self.space_dimension()))}
        return entity_dofs

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return numpy.prod(self.index_shape, dtype=int)

    @property
    def index_shape(self):
        ps = self._rule.point_set
        return tuple(index.extent for index in ps.indices)

    @property
    def value_shape(self):
        return ()

    @cached_property
    def fiat_equivalent(self):
        ps = self._rule.point_set
        if isinstance(ps, UnknownPointSet):
            raise ValueError("A quadrature element with rule with runtime points has no fiat equivalent!")
        weights = getattr(self._rule, 'weights', None)
        if weights is None:
            # we need the weights.
            weights, = evaluate([self._rule.weight_expression])
            weights = weights.arr.flatten()
            self._rule.weights = weights

        return FIAT.QuadratureElement(self.cell, ps.points, weights)

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''
        if entity is not None and entity != (self.cell.get_dimension(), 0):
            raise ValueError('QuadratureElement does not "tabulate" on subentities.')

        if order:
            raise ValueError("Derivatives are not defined on a QuadratureElement.")

        if not self._rule.point_set.almost_equal(ps):
            raise ValueError("Mismatch of quadrature points!")

        # Return an outer product of identity matrices
        multiindex = self.get_indices()
        product = reduce(gem.Product, [gem.Delta(q, r)
                                       for q, r in zip(ps.indices, multiindex)])

        dim = self.cell.get_spatial_dimension()
        return {(0,) * dim: gem.ComponentTensor(product, multiindex)}

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError("QuadratureElement cannot do point evaluation!")

    @property
    def dual_basis(self):
        ps = self._rule.point_set
        multiindex = self.get_indices()
        # Evaluation matrix is just an outer product of identity
        # matrices, evaluation points are just the quadrature points.
        Q = reduce(gem.Product, (gem.Delta(q, r)
                                 for q, r in zip(ps.indices, multiindex)))
        Q = gem.ComponentTensor(Q, multiindex)
        return Q, ps

    @property
    def mapping(self):
        return MappingStr("affine")
