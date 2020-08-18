from abc import ABCMeta, abstractproperty, abstractmethod
from functools import reduce
from itertools import chain

import numpy

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.quadrature import make_quadrature


class FiniteElementBase(metaclass=ABCMeta):

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
        return {dim: {e: sorted(chain(*[entity_dofs[d][se]
                                        for d, se in sub_entities]))
                      for e, sub_entities in entities.items()}
                for dim, entities in self.cell.sub_entities.items()}

    def entity_closure_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite
        element.'''
        return self._entity_closure_dofs

    @cached_property
    def _entity_support_dofs(self):
        esd = {}
        for entity_dim in self.cell.sub_entities.keys():
            beta = self.get_indices()
            zeta = self.get_value_indices()

            entity_cell = self.cell.construct_subelement(entity_dim)
            quad = make_quadrature(entity_cell, (2*numpy.array(self.degree)).tolist())

            eps = 1.e-8  # Is this a safe value?

            result = {}
            for f in self.entity_dofs()[entity_dim].keys():
                # Tabulate basis functions on the facet
                vals, = self.basis_evaluation(0, quad.point_set, entity=(entity_dim, f)).values()
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

            esd[entity_dim] = result
        return esd

    def entity_support_dofs(self):
        '''Return the map of topological entities to degrees of
        freedom that have non-zero support on those entities for the
        finite element.'''
        return self._entity_support_dofs

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

    @property
    def fiat_equivalent(self):
        '''The FIAT element equivalent to this FInAT element.'''
        raise NotImplementedError(
            f"Cannot make equivalent FIAT element for {type(self).__name__}"
        )

    def get_indices(self):
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        return tuple(gem.Index(extent=d) for d in self.index_shape)

    def get_value_indices(self):
        '''A tuple of GEM :class:`~gem.Index` of the correct extents to loop over
        the value shape of this element.'''

        return tuple(gem.Index(extent=d) for d in self.value_shape)

    @abstractmethod
    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        :param coordinate_mapping: a
           :class:`~.physically_mapped.PhysicalGeometry` object that
           provides physical geometry callbacks (may be None).
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
    def dual_basis(self):
        '''Returns a tuple where each element of the tuple represents one
        functional in the dual space. Each functional is represented by
        a tuple of tuples containing the points (PointSet), a weight tensor
        which holds the weights for each component of value_shape of functions,
        and a alpha tensor for extracting the alpha components from ReferenceGrad,
        sorted by total derivative order.

        For example, a dual basis containing 2 functionals with maximum derivative
        order of 1 would be represented by:

        .. highlight:: python
        .. code-block:: python

            ((((point_set_10, weight_tensor_10, alpha_tensor_10),
               (point_set_11, weight_tensor_11, alpha_tensor_11)), None)
             (((point_set_20, weight_tensor_20, alpha_tensor_20),
              ()), None)

        where one of the innermost tuples is empty because there are no evaluations
        at that order. The `None` is either a placeholder for \'normal\' elements,
        or a tuple containing the index of expression to extract using gem.Delta
        (currently just used by TensorFiniteElement).
        '''
        # TODO: Add label for type of evaluation?

    def dual_evaluation(self, fn, entity=None):
        '''Return code for performing the dual evaluation at the nodes of the
        reference element. Currently only works for non-derivatives.

        :param fn: Callable that takes in PointSet and returns GEM expression.
        :param entity: the cell entity on which to tabulate for comparing
                       results with FIAT.
        '''
        if entity is not None:
            # TODO: Add comparison to FIAT
            raise NotImplementedError("Comparison with FIAT is not implemented!")

        dual_expressions = []   # One for each functional
        expr_cache = {}         # Sharing of evaluation of the expression at points

        for dual, tensorfe_idx in self.dual_basis:
            qexprs = gem.Zero()
            for derivative_order, deriv in enumerate(dual):
                for tups in deriv:
                    try:
                        point_set, weight_tensor, alpha_tensor = tups
                    except ValueError:  # Empty
                        continue

                    try:
                        # TODO: Choose hash method
                        expr = expr_cache[(point_set.points.data.tobytes(), alpha_tensor.array.tobytes())]
                    except KeyError:
                        expr_grad = fn(point_set, derivative=derivative_order)
                        # TODO: Multiple alpha at once
                        if derivative_order == 0:
                            expr = expr_grad
                        else:
                            # Extract derivative component
                            alpha_idx = tuple(gem.Index(extent=fn.dimension) for _ in range(derivative_order))

                            # gem.partial_indexed but from back
                            rank = len(expr_grad.shape) - len(alpha_idx)
                            shape_indices = tuple(gem.Index() for _ in range(rank))
                            expr_partial = gem.ComponentTensor(
                                gem.Indexed(expr_grad, shape_indices + alpha_idx),
                                shape_indices)

                            expr = gem.index_sum(expr_partial * alpha_tensor[alpha_idx], alpha_idx)
                        expr_cache[(point_set.points.data.tobytes(), alpha_tensor.array.tobytes())] = expr

                    # Apply weights
                    # For point_set with multiple points
                    zeta = tuple(idx for _ in range(len(point_set.points)) for idx in self.get_value_indices())
                    if tensorfe_idx is None:
                        qexpr = gem.index_sum(gem.partial_indexed(expr, zeta) * weight_tensor[zeta], point_set.indices+zeta)
                    else:
                        deltas = reduce(gem.Product, (gem.Delta(z, t) for z, t in zip(zeta, tensorfe_idx)))
                        base_rank = len(self.value_shape) - len(tensorfe_idx)
                        # TODO: Fix non-scalar _base_element
                        qexpr = gem.index_sum(gem.partial_indexed(expr, zeta) * deltas * weight_tensor[tensorfe_idx[:base_rank]],
                                              point_set.indices+zeta)
                    # Sum for all derivatives
                    qexprs = gem.Sum(qexprs, qexpr)

            assert qexprs.shape == ()
            dual_expressions.append(qexprs)
        ir_shape = gem.ListTensor(dual_expressions)

        return ir_shape

    @abstractproperty
    def mapping(self):
        '''Appropriate mapping from the reference cell to a physical cell for
        all basis functions of the finite element.'''


def entity_support_dofs(elem, entity_dim):
    '''Return the map of entity id to the degrees of freedom for which
    the corresponding basis functions take non-zero values.

    :arg elem: FInAT finite element
    :arg entity_dim: Dimension of the cell subentity.
    '''
    return elem.entity_support_dofs()[entity_dim]
