from functools import partial
from operator import add, methodcaller

import numpy

import FIAT

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.point_set import ConcatPointSet


class EnrichedElement(FiniteElementBase):
    """A finite element whose basis functions are the union of the
    basis functions of several other finite elements."""

    def __new__(cls, elements):
        elements = tuple(elements)
        if len(elements) == 1:
            return elements[0]
        else:
            self = super().__new__(cls)
            self.elements = elements
            return self

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
        return concatenate_entity_dofs(self.cell, self.elements,
                                       methodcaller("entity_dofs"))

    @cached_property
    def _entity_support_dofs(self):
        return concatenate_entity_dofs(self.cell, self.elements,
                                       methodcaller("entity_support_dofs"))

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

    @cached_property
    def fiat_equivalent(self):
        # Avoid circular import dependency
        from finat.mixed import MixedSubElement

        if all(isinstance(e, MixedSubElement) for e in self.elements):
            # EnrichedElement is actually a MixedElement
            return FIAT.MixedElement([e.element.fiat_equivalent
                                      for e in self.elements], ref_el=self.cell)
        else:
            return FIAT.EnrichedElement(*(e.fiat_equivalent
                                          for e in self.elements))

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

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''
        results = [element.basis_evaluation(order, ps, entity, coordinate_mapping=coordinate_mapping)
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

    @property
    def dual_basis(self):
        xs = []
        splits = []
        exprs = []
        zetas = self.get_value_indices()
        for e in self.elements:
            Q, x = e.dual_basis
            alphas = e.get_indices()
            Q = gem.Indexed(Q, alphas + zetas)
            Q = gem.ComponentTensor(Q, alphas + x.indices)
            splits.append(len(alphas))
            xs.append(x)
            exprs.append(Q)
        Q = gem.TensorConcat(splits, *exprs)
        x = ConcatPointSet(xs)
        alphas = self.get_indices()
        Q = gem.Indexed(Q, alphas + x.indices)
        Q = gem.ComponentTensor(Q, alphas + zetas)
        return Q, x

    # def dual_evaluation(self, fn):
    #     exprs = []
    #     zetas = self.get_value_indices()
    #     psis = []
    #     splits = []
    #     for elem in self.elements:
    #         expr, alphas, psi, zeta = elem.dual_evaluation(fn)
    #         missing = tuple(i for i in alphas if i not in expr.free_indices)
    #         # Broadcast across missing free indices so that the
    #         # correct shape appears for concatenation.
    #         if missing:
    #             shape = tuple(i.extent for i in missing)
    #             expr = gem.Indexed(
    #                 gem.ListTensor(
    #                     numpy.asarray(
    #                         [expr for _ in
    #                          range(numpy.prod(shape, dtype=int))],
    #                         dtype=object
    #                     ).reshape(shape)
    #                 ),
    #                 missing
    #             )
    #         assert len(zeta) == len(zetas)
    #         expr = gem.ComponentTensor(
    #             gem.Indexed(
    #                 gem.ComponentTensor(expr, alphas + psi + zeta),
    #                 alphas + psi + zetas
    #             ),
    #             alphas + psi
    #         )
    #         splits.append(len(alphas))
    #         exprs.append(expr)
    #         psis.append(psi)
    #     expr = gem.TensorConcat(tuple(splits), *exprs)
    #     alphas = self.get_indices()
    #     psi = (gem.Index(),)
    #     expr = gem.Indexed(expr, alphas + psi)
    #     return expr, alphas, psi, zetas

    @property
    def dual_point_set(self):
        return ConcatPointSet([e.dual_point_set for e in self.elements])

    @property
    def mapping(self):
        mappings = set(elem.mapping for elem in self.elements)
        if len(mappings) != 1:
            return None
        else:
            result, = mappings
            return result


def tree_map(f, *args):
    """Like the built-in :py:func:`map`, but applies to a tuple tree."""
    nonleaf, = set(isinstance(arg, tuple) for arg in args)
    if nonleaf:
        ndim, = set(map(len, args))  # asserts equal arity of all args
        return tuple(tree_map(f, *subargs) for subargs in zip(*args))
    else:
        return f(*args)


def concatenate_entity_dofs(ref_el, elements, method):
    """Combine the entity DoFs from a list of elements into a combined
    dict containing the information for the concatenated DoFs of all
    the elements.

    :arg ref_el: the reference cell
    :arg elements: subelement whose DoFs are concatenated
    :arg method: method to obtain the entity DoFs dict
    :returns: concatenated entity DoFs dict
    """
    entity_dofs = {dim: {i: [] for i in entities}
                   for dim, entities in ref_el.get_topology().items()}
    offsets = numpy.cumsum([0] + list(e.space_dimension()
                                      for e in elements), dtype=int)
    for i, d in enumerate(map(method, elements)):
        for dim, dofs in d.items():
            for ent, off in dofs.items():
                entity_dofs[dim][ent] += list(map(partial(add, offsets[i]), off))
    return entity_dofs
