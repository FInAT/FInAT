from abc import ABCMeta, abstractproperty, abstractmethod
from functools import reduce
from itertools import chain

import numpy

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.quadrature import make_quadrature
from finat.point_set import PointSet, PointSingleton

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
              ()), None))

        where one of the innermost tuples is empty because there are no evaluations
        at that order. The `None` is either a placeholder for \'normal\' elements,
        or a tuple containing the index of expression to extract using gem.Delta
        (currently just used by TensorFiniteElement).
        '''
        # TODO: Add label for type of evaluation?

    def dual_evaluation(self, fn, entity=None):
        '''Return code for performing the dual basis evaluation at the nodes of
        the reference element. Currently only works for non-derivatives.

        :param fn: Callable for point evaluation of the expression to dual
                   evaluate. The callable should take in a PointSet and return
                   a GEM expression for the point evaluation of the expression
                   at those points. Requires a dimension attribute storing
                   topological dimension.
        :param entity: the cell entity on which to tabulate for comparing
                       results with FIAT.
        :returns: A gem tensor with (num_nodes,) shape and any number of free
                  indices.
        '''
        if entity is not None:
            # TODO: Add comparison to FIAT
            raise NotImplementedError("Comparison with FIAT is not implemented!")

        if any(len(dual) > 1 for dual, tensorfe_idx in self.dual_basis):
            raise NotImplementedError("Can only interpolate onto dual basis functionals"
                                      " without derivative evaluation, sorry!")

        dual_expressions = []   # One for each functional
        expr_cache = {}         # Sharing of evaluation of the expression at points

        # A tensor of weights (of total rank R) to contract with a unique
        # vector of points to evaluate at, giving a tensor (of total rank R-1)
        # where the first indices (rows) correspond to a basis functional
        # (node).
        # DOK Sparse matrix in (row, col, higher,..)=>value pairs - to become a
        # gem.SparseLiteral.
        # Rows are number of nodes/dual functionals.
        # Columns are unique points to evaluate.
        # Higher indices are tensor indices of the weights when weights are
        # tensor valued.
        Q = {}

        # List of unique points to evaluate in the order required for correct
        # contraction with Q - to become a FInAT.PointSet
        x = []

        #
        # BUILD Q MATRIX
        #

        # FIXME: The below loop is REALLY SLOW for BDM - Q and x should just be output as the dual basis

        can_construct = True
        last_shape = None
        # i are rows of Q
        for i in range(len(self.dual_basis)):
            # Ignore tensorfe_idx
            dual_functional_w_derivs, tensor_idx = self.dual_basis[i]
            # Can only build if not tensor valued
            if tensor_idx is not None:
                can_construct = False
                break
            # Only use 1st entry in dual which assumes no derivatives otherwise
            # try other method.
            if len(dual_functional_w_derivs) != 1:
                can_construct = False
                break
            dual_functional = dual_functional_w_derivs[0]
            for j in range(len(dual_functional)):
                # Ignore alpha, just extract point (x_j) and weights (q_j)
                x_j, q_j, _ = dual_functional[j]

                # Esure all weights have the same shape
                if last_shape is not None:
                    assert q_j.shape == last_shape
                last_shape = q_j.shape

                assert q_j.children == ()
                assert q_j.free_indices == ()

                # Get k - the columns of Q.
                x_matches = [x_j.almost_equal(x_k) for x_k in x]
                if any(x_matches):
                    k = x_matches.index(True)
                else:
                    k = len(x)
                    x.append(x_j)

                # q_j may be tensor valued
                it = numpy.nditer(q_j.array, flags=['multi_index'])
                for q_j_entry in it:
                    Q[(i, k) + it.multi_index] = q_j_entry

        if can_construct:

            #
            # CONVERT Q TO gem.SparseLiteral
            #

            # Convert dictionary of keys to SparseLiteral
            Q = gem.SparseLiteral(Q) # FIXME: This fails to compile since Impero can't yet deal with a sparse tensor
            # FIXME: Temporarily use a normal literal
            Q = gem.Literal(Q.array.todense())

            #
            # CONVERT x TO gem.PointSet
            #

            # Convert list of points to equivalent PointSet
            allpts = None
            dim = None
            for i in range(len(x)):
                # For the future - can only have one UnknownPointSingleton in the
                # pointset.
                # if isinstance(x[i], UnknownPointSingleton):
                #     assert len(x) == 1
                #     x = x[0]
                # and skip x = PointSet(allpts)
                if dim is not None:
                    assert x[i].dimension == dim
                dim = x[i].dimension
                if allpts is not None:
                    allpts = numpy.concatenate((allpts, x[i].points), axis=0)
                else:
                    allpts = x[i].points
            assert allpts.shape[1] == dim
            x = PointSet(allpts)

            #
            # EVALUATE fn AT x
            #
            expr = fn(x)

            #
            # TENSOR CONTRACT Q WITH expr
            #
            expr_shape_indices = tuple(gem.Index(extent=ex) for ex in expr.shape)
            expr_arg_indices = tuple(set(expr.free_indices) - set(x.indices))
            assert Q.free_indices == ()
            Q2_shape_indices = tuple(gem.Index(extent=ex) for ex in Q.shape)
            assert tuple(i.extent for i in Q2_shape_indices[2:]) == tuple(i.extent for i in expr_shape_indices)
            basis_indices = Q2_shape_indices[:1]
            dual_eval_is = gem.index_sum(Q[basis_indices + x.indices + expr_shape_indices] * expr[expr_shape_indices], x.indices+expr_shape_indices)
            # Need to only have node count as free indices
            dual_eval_is_w_shape = gem.ComponentTensor(dual_eval_is, basis_indices)
            assert dual_eval_is_w_shape.shape[0] == Q.shape[0]
            return dual_eval_is_w_shape

        else: # Can't construct Q so use old method

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
                        if tensorfe_idx is None:
                            zeta = tuple(idx for _ in range(len(point_set.points)) for idx in self.get_value_indices())
                            qexpr = gem.index_sum(gem.partial_indexed(expr, zeta) * weight_tensor[zeta], point_set.indices + zeta)
                        else:
                            base_rank = len(self.value_shape) - len(tensorfe_idx)

                            zeta_base = tuple(idx for _ in range(len(point_set.points)) for idx in
                                            [gem.Index(extent=d)for d in self.value_shape[:base_rank]])
                            zeta_tensor = tuple(idx for _ in range(len(point_set.points)) for idx in
                                                [gem.Index(extent=d)for d in self.value_shape[base_rank:]])
                            deltas = reduce(gem.Product, (gem.Delta(z, t) for z, t in zip(zeta_tensor, tensorfe_idx)))
                            zeta = zeta_tensor + zeta_base

                            qexpr = gem.index_sum(gem.partial_indexed(expr, zeta) * deltas * weight_tensor[zeta_base],
                                                point_set.indices + zeta)
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
