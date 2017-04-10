from __future__ import absolute_import, print_function, division
from six import iteritems
from six.moves import range, zip
from functools import reduce

import numpy

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.quadrature import make_quadrature


class QuadratureElement(FiniteElementBase):
    """A set of quadrature points pretending to be a finite element."""

    def __init__(self, cell, degree, scheme="default"):
        self.cell = cell
        self._rule = make_quadrature(cell, degree, scheme)

    @cached_property
    def cell(self):
        pass  # set at initialisation

    @property
    def degree(self):
        raise NotImplementedError("QuadratureElement does not represent a polynomial space.")

    @cached_property
    def _entity_dofs(self):
        # Inspired by ffc/quadratureelement.py
        entity_dofs = {dim: {entity: [] for entity in entities}
                       for dim, entities in iteritems(self.cell.get_topology())}
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

    def basis_evaluation(self, order, ps, entity=None):
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
