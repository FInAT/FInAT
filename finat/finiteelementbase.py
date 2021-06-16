from abc import ABCMeta, abstractproperty, abstractmethod
from functools import reduce
from itertools import chain

import numpy

import gem
from gem.interpreter import evaluate
from gem.utils import cached_property

from finat.quadrature import make_quadrature
from finat.point_set import PointSet


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
        '''Base method which will work in most cases but can be
        overwritten as necessary'''
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

        # Dict of unique points to evaluate stored as
        # {hash(tuple(pt_hash.flatten())): (x_k, k)} pairs. Points in index k
        # order form a vector required for correct contraction with Q. Will
        # become a FInAT.PointSet later.
        # pt_hash = numpy.round(x_k, x_hash_decimals) such that
        # numpy.allclose(x_k, pt_hash, atol=1*10**-dec) == true
        x = {}
        x_hash_decimals = 12
        x_hash_atol = 1e-12  # = 1*10**-dec

        #
        # BUILD Q TENSOR
        #

        # FIXME: The below loop is REALLY SLOW for BDM - Q and x should just be output as the dual basis

        can_construct = True
        last_shape = None
        self.Q_is_identity = True  # TODO do this better

        dual_basis_tuple = fiat_element_dual_basis(self._element)

        # i are rows of Q
        for i in range(len(dual_basis_tuple)):
            # Ignore tensorfe_idx
            dual_functional_w_derivs, tensor_idx = dual_basis_tuple[i]
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

                # Create hash into x
                x_hash = numpy.round(x_j.points, x_hash_decimals)
                assert numpy.allclose(x_j.points, x_hash, atol=x_hash_atol)
                x_hash = hash(tuple(x_hash.flatten()))

                # Get value and index k or add to dict. k are the columns of Q.
                try:
                    x_j, k = x[x_hash]
                except KeyError:
                    k = len(x)
                    x[x_hash] = x_j, k

                # q_j may be tensor valued
                it = numpy.nditer(q_j.array, flags=['multi_index'])
                for q_j_entry in it:
                    Q[(i, k) + it.multi_index] = q_j_entry
                    if len(set((i, k) + it.multi_index)) > 1:
                        # Identity has i == k == it.multi_index[0] == ...
                        # Since i increases from 0 in increments of 1 we know
                        # that if this check is not triggered we definitely
                        # have an identity tensor.
                        self.Q_is_identity = False

        if not can_construct:
            raise NotImplementedError("todo!")

        #
        # CONVERT Q TO gem.Literal (TODO: should be a sparse tensor)
        #

        # temporary until sparse literals are implemented in GEM which will
        # automatically convert a dictionary of keys internally.
        import sparse
        Q = gem.Literal(sparse.as_coo(Q).todense())

        #
        # CONVERT x TO gem.PointSet
        #

        # Convert PointSets to a single PointSet with the correct ordering
        # for contraction with Q
        random_pt, _ = next(iter(x.values()))
        dim = random_pt.dimension
        allpts = numpy.empty((len(x), dim), dtype=random_pt.points.dtype)
        for _, (x_k, k) in x.items():
            assert x_k.dimension == dim
            allpts[k, :] = x_k.points
            # FIXME
            # For the future - can only have one UnknownPointSingleton in the
            # pointset.
            # if isinstance(x[i], UnknownPointSingleton):
            #     assert len(x) == 1
            #     x = x[0]
            # and skip x = PointSet(allpts)
        assert allpts.shape[1] == dim
        x = PointSet(allpts)
        return (Q, x)

    def dual_evaluation(self, fn):
        '''Return code for performing the dual basis evaluation at the nodes of
        the reference element. Currently only works for non-derivatives.

        :param fn: Callable representing the function to dual evaluate.
                   Callable should take in an :class:`AbstractPointSet` and
                   return a GEM expression for evaluation of the function at
                   those points. If the callable provides a ``.factors``
                   property then it may be used for sum factorisation in
                   :class:`TensorProductElement`s
        :returns: A gem tensor with (num_nodes,) shape and any number of free
                  indices.

        Base method which will work in most cases but can be overwritten as
        necessary.'''

        Q, x = self.dual_basis

        #
        # EVALUATE fn AT x
        #
        expr = fn(x)

        #
        # TENSOR CONTRACT Q WITH expr
        #
        expr_shape_indices = tuple(gem.Index(extent=ex) for ex in expr.shape)
        assert Q.free_indices == ()
        Q_shape_indices = tuple(gem.Index(extent=ex) for ex in Q.shape)
        assert tuple(i.extent for i in Q_shape_indices[2:]) == tuple(i.extent for i in expr_shape_indices)
        basis_indices = Q_shape_indices[:1]
        if self.Q_is_identity and expr.free_indices != ():
            assert len(set(Q.shape)) == 1
            # Don't bother multiplying by an identity tensor

            # FIXME - Since expr can have no free indices at this
            # point (see TSFC issue #240), there's no easy way to make this
            # short cut where expr.free_indices == () whilst maintaining
            # the interface that dual_evaluation returns something with
            # (num_nodes,) shape. To make this work, I'll need to change
            # driver.py to expect a different interface.

            dual_eval_is = expr
            # replace the free index with an index of the same extent in
            # expr. TODO: Consider if basis_indices can be found in Q in
            # general by checking index extents
            basis_index = tuple(i for i in expr.free_indices if i.extent == basis_indices[0].extent)[0]
            basis_indices = (basis_index,)
        else:
            dual_eval_is = gem.optimise.make_product((Q[basis_indices + x.indices + expr_shape_indices], expr[expr_shape_indices]), x.indices+expr_shape_indices)
        dual_eval_is_w_shape = gem.ComponentTensor(dual_eval_is, basis_indices)
        assert dual_eval_is_w_shape.shape[0] == Q.shape[0]
        return dual_eval_is_w_shape

    @abstractproperty
    def mapping(self):
        '''Appropriate mapping from the reference cell to a physical cell for
        all basis functions of the finite element.'''


# TODO update this
def fiat_element_dual_basis(fiat_element):

    max_deriv_order = max([ell.max_deriv_order for ell in fiat_element.dual_basis()])

    duals = []
    point_set_cache = {}    # To avoid repeating points?
    for dual in fiat_element.dual_basis():
        derivs = []
        pts_in_derivs = []

        # For non-derivatives
        for pt, tups in sorted(dual.get_point_dict().items()):  # Ensure parallel safety
            try:
                point_set = point_set_cache[(pt,)]
            except KeyError:
                point_set = PointSet((pt,))
                point_set_cache[(pt,)] = point_set

            # No derivative component extraction required
            alpha_tensor = gem.Literal(1)

            weight_dict = {c: w for w, c in tups}
            # Each entry of tensor is weight of that component
            if len(fiat_element.value_shape()) == 0:
                weight_tensor = gem.Literal(weight_dict[tuple()])
            else:
                weight_array = np.zeros(fiat_element.value_shape())
                for key, item in weight_dict.items():
                    weight_array[key] = item
                weight_tensor = gem.Literal(weight_array)

            pts_in_derivs.append((point_set, weight_tensor, alpha_tensor))
        derivs.append(pts_in_derivs)

        # For derivatives
        deriv_dict_items = sorted(dual.deriv_dict.items())  # Ensure parallel safety
        for derivative_order in range(1, max_deriv_order+1):
            pts_in_derivs = []
            # TODO: Combine tensors for tups of same derivative order
            for pt, tups in deriv_dict_items:
                weights, alphas, cmps = zip(*tups)

                # TODO: Repeated points and repeated tups
                # Get evaluations of this derivative order
                weight, alpha, cmp = [], [], []
                for j, a in enumerate(alphas):
                    if sum(a) == derivative_order:
                        weight.append(weights[j])
                        alpha.append(alphas[j])
                        cmp.append(cmps[j])
                if len(alpha) == 0:
                    continue

                # alpha_tensor assumes weights for each derivative component are equal
                # TODO: Case for unequal weights
                if len(alpha) > 1:
                    if 0.0 not in weight:
                        assert np.isclose(min(weight), max(weight))
                    else:
                        non_zero_weight = [w for w in weight if w != 0.0]
                        assert np.isclose(min(non_zero_weight), max(non_zero_weight))

                # For direct indexing
                alpha_idx = tuple(tuple(chain(*[(j,)*a for j, a in enumerate(alph)])) for alph in alpha)
                # TODO: How to ensure correct combination (indexing) with weight_tensor?
                alpha_arr = np.zeros((fiat_element.cell().get_spatial_dimension(),) * derivative_order)
                for idx in alpha_idx:
                    alpha_arr[idx] = 1
                alpha_tensor = gem.Literal(alpha_arr)

                try:
                    point_set = point_set_cache[(pt,)]
                except KeyError:
                    point_set = PointSet((pt,))
                    point_set_cache[(pt,)] = point_set

                weight_dict = dict(zip(cmp, weight))
                # Each entry of tensor is weight of that component
                if len(fiat_element.value_shape()) == 0:
                    weight_tensor = gem.Literal(weight_dict[tuple()])
                else:
                    weight_array = np.zeros(fiat_element.value_shape())
                    for key, item in weight_dict.items():
                        weight_array[key] = item
                    weight_tensor = gem.Literal(weight_array)

                pts_in_derivs.append((point_set, weight_tensor, alpha_tensor))
            derivs.append(tuple(pts_in_derivs))
        duals.append(tuple([tuple(derivs), None]))
    return tuple(duals)


def entity_support_dofs(elem, entity_dim):
    '''Return the map of entity id to the degrees of freedom for which
    the corresponding basis functions take non-zero values.

    :arg elem: FInAT finite element
    :arg entity_dim: Dimension of the cell subentity.
    '''
    return elem.entity_support_dofs()[entity_dim]
