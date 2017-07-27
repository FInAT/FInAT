from __future__ import absolute_import, print_function, division
from six import with_metaclass, iteritems, itervalues

from abc import ABCMeta, abstractproperty, abstractmethod
from itertools import chain

import numpy

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.quadrature import make_quadrature


class FiniteElementBase(with_metaclass(ABCMeta)):

    @abstractproperty
    def cell(self):
        '''The reference cell on which the element is defined.'''

    @abstractproperty
    def degree(self):
        '''The degree of the embedding polynomial space.

        In the tensor case this is a tuple.
        '''

    @abstractproperty
    def formdegree(self):
        '''Degree of the associated form (FEEC)'''

    @abstractmethod
    def entity_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom for the finite element.'''

    @cached_property
    def _entity_closure_dofs(self):
        # Compute the nodes on the closure of each sub_entity.
        entity_dofs = self.entity_dofs()
        return {dim: {e: list(chain(*[entity_dofs[d][se]
                                      for d, se in sub_entities]))
                      for e, sub_entities in iteritems(entities)}
                for dim, entities in iteritems(self.cell.sub_entities)}

    def entity_closure_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite
        element.'''
        return self._entity_closure_dofs

    @abstractmethod
    def space_dimension(self):
        '''Return the dimension of the finite element space.'''

    @abstractproperty
    def index_shape(self):
        '''A tuple indicating the number of degrees of freedom in the
        element. For example a scalar quadratic Lagrange element on a triangle
        would return (6,) while a vector valued version of the same element
        would return (6, 2)'''

    @abstractproperty
    def value_shape(self):
        '''A tuple indicating the shape of the element.'''

    def get_indices(self):
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        return tuple(gem.Index(extent=d) for d in self.index_shape)

    def get_value_indices(self):
        '''A tuple of GEM :class:`~gem.Index` of the correct extents to loop over
        the value shape of this element.'''

        return tuple(gem.Index(extent=d) for d in self.value_shape)

    @abstractmethod
    def basis_evaluation(self, order, ps, entity=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''

    @abstractmethod
    def point_evaluation(self, order, refcoords, entity=None):
        '''Return code for evaluating the element at an arbitrary points on
        the reference element.

        :param order: return derivatives up to this order.
        :param refcoords: GEM expression representing the coordinates
                          on the reference entity.  Its shape must be
                          a vector with the correct dimension, its
                          free indices are arbitrary.
        :param entity: the cell entity on which to tabulate.
        '''

    @abstractproperty
    def mapping(self):
        '''Appropriate mapping from the reference cell to a physical cell for
        all basis functions of the finite element.'''


def entity_support_dofs(elem, entity_dim):
    """Return the map of entity id to the degrees of freedom for which
    the corresponding basis functions take non-zero values.

    :arg elem: FInAT finite element
    :arg entity_dim: Dimension of the cell subentity.
    """
    if not hasattr(elem, "_entity_support_dofs"):
        elem._entity_support_dofs = {}
    cache = elem._entity_support_dofs
    try:
        return cache[entity_dim]
    except KeyError:
        pass

    beta = elem.get_indices()
    zeta = elem.get_value_indices()

    entity_cell = elem.cell.construct_subelement(entity_dim)
    quad = make_quadrature(entity_cell, (2*numpy.array(elem.degree)).tolist())

    eps = 1.e-8  # Is this a safe value?

    result = {}
    for f in elem.entity_dofs()[entity_dim].keys():
        # Tabulate basis functions on the facet
        vals, = itervalues(elem.basis_evaluation(0, quad.point_set, entity=(entity_dim, f)))
        # Integrate the square of the basis functions on the facet.
        ints = gem.IndexSum(
            gem.Product(gem.IndexSum(gem.Product(gem.Indexed(vals, beta + zeta),
                                                 gem.Indexed(vals, beta + zeta)), zeta),
                        quad.weight_expression),
            quad.point_set.indices
        )
        evaluation, = evaluate([gem.ComponentTensor(ints, beta)])
        ints = evaluation.arr.flatten()
        assert evaluation.fids == ()
        result[f] = [dof for dof, i in enumerate(ints) if i > eps]

    cache[entity_dim] = result
    return result
