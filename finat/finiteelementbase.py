from abc import ABCMeta, abstractproperty, abstractmethod
from itertools import chain

import numpy

import gem
from gem.interpreter import evaluate
from gem.optimise import delta_elimination, sum_factorise, traverse_product
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

    @property
    def dual_basis(self):
        '''Return a dual evaluation gem weight tensor Q and point set x to dual
        evaluate a function fn at.

        The general dual evaluation is then Q * fn(x).

        Note that the contraction index of the point set x is indexed out of Q
        to avoid confusion when trying index it out of Q later in
        dual_evaluation.

        If the dual weights are scalar then Q, for a general scalar FIAT
        element, is a matrix with dimensions
        (num_nodes, num_points)
        where num_points is made a free index that match the free index of x.

        If the dual weights are tensor valued then Q, for a general tensor
        valued FIAT element, is a tensor with dimensions
        (num_nodes, num_points, dual_weight_shape[0], ..., dual_weight_shape[n])
        where num_points made a free index that matches the free index of x.

        If the dual basis is of a tensor product or FlattenedDimensions element
        with N factors then Q in general is a tensor with dimensions
        (num_nodes_factor1, ..., num_nodes_factorN,
            num_points_factor1, ..., num_points_factorN,
            dual_weight_shape[0], ..., dual_weight_shape[n])
        where num_points_factorX are made free indices that match the free
        indices of x (which is now a TensorPointSet).

        If the dual basis is of a tensor finite element with some shape
        (S1, S2, ..., Sn) then the tensor element tQ is constructed from the
        base element's Q by taking the outer product with appropriately sized
        identity matrices:
        tQ = Q ⊗ 𝟙ₛ₁ ⊗ 𝟙ₛ₂ ⊗ ... ⊗ 𝟙ₛₙ
        '''
        raise NotImplementedError(
            f"Dual basis not defined for element {type(self).__name__}"
        )

    def dual_evaluation(self, fn):
        '''Return code for performing the dual basis evaluation at the nodes of
        the reference element. Currently only works for non-derivatives (not
        implemented) and flat elements (implemented in TensorFiniteElement).

        :param fn: Callable representing the function to dual evaluate.
                   Callable should take in an :class:`AbstractPointSet` and
                   return a GEM expression for evaluation of the function at
                   those points.
        :returns: A tuple (dual_evaluation_indexed_sum, basis_indices)
        '''
        Q, x = self.dual_basis

        expr = fn(x)
        # Apply targeted sum factorisation and delta elimination to
        # the expression
        sum_indices, factors = delta_elimination(*traverse_product(expr))
        expr = sum_factorise(sum_indices, factors)
        # NOTE: any shape indices in the expression are because the
        # expression is tensor valued.
        assert expr.shape == Q.shape[len(Q.shape)-len(expr.shape):]
        shape_indices = gem.indices(len(expr.shape))
        basis_indices = gem.indices(len(Q.shape) - len(expr.shape))
        Qi = Q[basis_indices + shape_indices]
        expri = expr[shape_indices]
        evaluation = gem.IndexSum(Qi * expri, x.indices + shape_indices)
        # Now we want to factorise over the new contraction with x,
        # ignoring any shape indices (this is a bit of a hack)
        # Really need to do a more targeted job here.
        evaluation = gem.optimise.contraction(evaluation, shape_indices)
        return evaluation, basis_indices

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
