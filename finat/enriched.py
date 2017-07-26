from __future__ import absolute_import, print_function, division
from six.moves import map, zip

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase


class EnrichedElement(FiniteElementBase):
    """A finite element whose basis functions are the union of the
    basis functions of several other finite elements."""

    def __init__(self, elements):
        super(EnrichedElement, self).__init__()
        self.elements = tuple(elements)

    @cached_property
    def cell(self):
        result, = set(elem.cell for elem in self.elements)
        return result

    @cached_property
    def degree(self):
        return tree_map(max, *[elem.degree for elem in self.elements])

    @cached_property
    def formdegree(self):
        ks = set(elem.formdegree for elem in self.elements)
        if None in ks:
            return None
        else:
            return max(ks)

    def entity_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom for the finite element.'''
        from FIAT.mixed import concatenate_entity_dofs
        return concatenate_entity_dofs(self.cell, self.elements)

    def space_dimension(self):
        '''Return the dimension of the finite element space.'''
        return sum(elem.space_dimension() for elem in self.elements)

    @cached_property
    def index_shape(self):
        return (self.space_dimension(),)

    @cached_property
    def value_shape(self):
        '''A tuple indicating the shape of the element.'''
        shape, = set(elem.value_shape for elem in self.elements)
        return shape

    def _compose_evaluations(self, results):
        keys, = set(map(frozenset, results))

        def merge(tables):
            tables = tuple(tables)
            zeta = self.get_value_indices()
            tensors = []
            for elem, table in zip(self.elements, tables):
                beta_i = elem.get_indices()
                tensors.append(gem.ComponentTensor(
                    gem.Indexed(table, beta_i + zeta),
                    beta_i
                ))
            beta = self.get_indices()
            return gem.ComponentTensor(
                gem.Indexed(gem.Concatenate(*tensors), beta),
                beta + zeta
            )
        return {key: merge(result[key] for result in results)
                for key in keys}

    def basis_evaluation(self, order, ps, entity=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''
        results = [element.basis_evaluation(order, ps, entity)
                   for element in self.elements]
        return self._compose_evaluations(results)

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
        results = [element.point_evaluation(order, refcoords, entity)
                   for element in self.elements]
        return self._compose_evaluations(results)


def tree_map(f, *args):
    """Like the built-in :py:func:`map`, but applies to a tuple tree."""
    nonleaf, = set(isinstance(arg, tuple) for arg in args)
    if nonleaf:
        ndim, = set(map(len, args))  # asserts equal arity of all args
        return tuple(tree_map(f, *subargs) for subargs in zip(*args))
    else:
        return f(*args)
