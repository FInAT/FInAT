from functools import reduce
import itertools
from itertools import chain, product
from operator import methodcaller

import numpy

import FIAT
from FIAT.polynomial_set import mis
from FIAT.reference_element import TensorProductCell

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.point_set import PointSingleton, PointSet, TensorPointSet


class TensorProductElement(FiniteElementBase):

    def __init__(self, factors):
        super(TensorProductElement, self).__init__()
        self.factors = tuple(factors)

        shapes = [fe.value_shape for fe in self.factors if fe.value_shape != ()]
        if len(shapes) == 0:
            self._value_shape = ()
        elif len(shapes) == 1:
            self._value_shape = shapes[0]
        else:
            raise NotImplementedError("Only one nonscalar factor permitted!")

    @cached_property
    def cell(self):
        return TensorProductCell(*[fe.cell for fe in self.factors])

    @property
    def degree(self):
        return tuple(fe.degree for fe in self.factors)

    @cached_property
    def formdegree(self):
        if any(fe.formdegree is None for fe in self.factors):
            return None
        else:
            return sum(fe.formdegree for fe in self.factors)

    @cached_property
    def _entity_dofs(self):
        return productise(self.factors, methodcaller("entity_dofs"))

    @cached_property
    def _entity_support_dofs(self):
        return productise(self.factors, methodcaller("entity_support_dofs"))

    def entity_dofs(self):
        return self._entity_dofs

    @cached_property
    def entity_permutations(self):
        return compose_permutations(self.factors)

    def space_dimension(self):
        return numpy.prod([fe.space_dimension() for fe in self.factors])

    @property
    def index_shape(self):
        return tuple(chain(*[fe.index_shape for fe in self.factors]))

    @property
    def value_shape(self):
        return self._value_shape

    @cached_property
    def fiat_equivalent(self):
        # FIAT TensorProductElement support only 2 factors
        A, B = self.factors
        return FIAT.TensorProductElement(A.fiat_equivalent, B.fiat_equivalent)

    def _factor_entity(self, entity):
        # Default entity
        if entity is None:
            entity = (self.cell.get_dimension(), 0)
        entity_dim, entity_id = entity

        # Factor entity
        assert isinstance(entity_dim, tuple)
        assert len(entity_dim) == len(self.factors)

        shape = tuple(len(c.get_topology()[d])
                      for c, d in zip(self.cell.cells, entity_dim))
        entities = list(zip(entity_dim, numpy.unravel_index(entity_id, shape)))
        return entities

    def _merge_evaluations(self, factor_results):
        # Spatial dimension
        dimension = self.cell.get_spatial_dimension()

        # Derivative order
        order = max(map(sum, chain(*factor_results)))

        # A list of slices that are used to select dimensions
        # corresponding to each subelement.
        dim_slices = TensorProductCell._split_slices([c.get_spatial_dimension()
                                                      for c in self.cell.cells])

        # A list of multiindices, one multiindex per subelement, each
        # multiindex describing the shape of basis functions of the
        # subelement.
        alphas = [fe.get_indices() for fe in self.factors]

        # A list of multiindices, one multiindex per subelement, each
        # multiindex describing the value shape of the subelement.
        zetas = [fe.get_value_indices() for fe in self.factors]

        result = {}
        for derivative in range(order + 1):
            for Delta in mis(dimension, derivative):
                # Split the multiindex for the subelements
                deltas = [Delta[s] for s in dim_slices]
                # GEM scalars (can have free indices) for collecting
                # the contributions from the subelements.
                scalars = []
                for fr, delta, alpha, zeta in zip(factor_results, deltas, alphas, zetas):
                    # Turn basis shape to free indices, select the
                    # right derivative entry, and collect the result.
                    scalars.append(gem.Indexed(fr[delta], alpha + zeta))
                # Multiply the values from the subelements and wrap up
                # non-point indices into shape.
                result[Delta] = gem.ComponentTensor(
                    reduce(gem.Product, scalars),
                    tuple(chain(*(alphas + zetas)))
                )
        return result

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        entities = self._factor_entity(entity)
        entity_dim, _ = zip(*entities)

        ps_factors = factor_point_set(self.cell, entity_dim, ps)

        factor_results = [fe.basis_evaluation(order, ps_, e)
                          for fe, ps_, e in zip(self.factors, ps_factors, entities)]

        return self._merge_evaluations(factor_results)

    def point_evaluation(self, order, point, entity=None):
        entities = self._factor_entity(entity)
        entity_dim, _ = zip(*entities)

        # Split point expression
        assert len(self.cell.cells) == len(entity_dim)
        point_dims = [cell.construct_subelement(dim).get_spatial_dimension()
                      for cell, dim in zip(self.cell.cells, entity_dim)]
        assert isinstance(point, gem.Node) and point.shape == (sum(point_dims),)
        slices = TensorProductCell._split_slices(point_dims)
        point_factors = []
        for s in slices:
            point_factors.append(gem.ListTensor(
                [gem.Indexed(point, (i,))
                 for i in range(s.start, s.stop)]
            ))

        # Subelement results
        factor_results = [fe.point_evaluation(order, p_, e)
                          for fe, p_, e in zip(self.factors, point_factors, entities)]

        return self._merge_evaluations(factor_results)

    @property
    def dual_basis(self):
        # Outer product the dual bases of the factors
        qs, pss = zip(*(factor.dual_basis for factor in self.factors))
        ps = TensorPointSet(pss)
        # Naming as _merge_evaluations above
        alphas = [factor.get_indices() for factor in self.factors]
        zetas = [factor.get_value_indices() for factor in self.factors]
        # Index the factors by so that we can reshape into index-shape
        # followed by value-shape
        qis = [q[alpha + zeta] for q, alpha, zeta in zip(qs, alphas, zetas)]
        Q = gem.ComponentTensor(
            reduce(gem.Product, qis),
            tuple(chain(*(alphas + zetas)))
        )
        return Q, ps

    @cached_property
    def mapping(self):
        mappings = [fe.mapping for fe in self.factors if fe.mapping != "affine"]
        if len(mappings) == 0:
            return "affine"
        elif len(mappings) == 1:
            return mappings[0]
        else:
            return None


def productise(factors, method):
    '''Tensor product the dict mapping topological entities to dofs across factors.

    :arg factors: element factors.
    :arg method: instance method to call on each factor to get dofs.'''
    shape = tuple(fe.space_dimension() for fe in factors)
    dofs = {}
    for dim in product(*[fe.cell.get_topology().keys()
                         for fe in factors]):
        dim_dofs = []
        topds = [method(fe)[d]
                 for fe, d in zip(factors, dim)]
        for tuple_ei in product(*[sorted(topd) for topd in topds]):
            tuple_vs = list(product(*[topd[ei]
                                      for topd, ei in zip(topds, tuple_ei)]))
            if tuple_vs:
                vs = list(numpy.ravel_multi_index(numpy.transpose(tuple_vs), shape))
                dim_dofs.append((tuple_ei, vs))
            else:
                dim_dofs.append((tuple_ei, []))
        # flatten entity numbers
        dofs[dim] = dict(enumerate(v for k, v in sorted(dim_dofs)))
    return dofs


def compose_permutations(factors):
    """For the :class:`TensorProductElement` object composed of factors,
    construct, for each dimension tuple, for each entity, and for each possible
    entity orientation combination, the DoF permutation list.

    :arg factors: element factors.
    :returns: entity_permutation dict of the :class:`TensorProductElement` object
        composed of factors.

    For tensor-product elements, one needs to consider two kinds of orientations:
    extrinsic orientations and intrinsic ("material") orientations.

    Example:

    UFCQuadrilateral := UFCInterval x UFCInterval

    eo (extrinsic orientation): swap axes (X -> y, Y-> x)
    io (intrinsic orientation): reflect component intervals
    o (total orientation)     : (2 ** dim) * eo + io

    eo\\io    0      1      2      3

           1---3  0---2  3---1  2---0
      0    |   |  |   |  |   |  |   |
           0---2  1---3  2---0  3---1

           2---3  3---2  0---1  1---0
      1    |   |  |   |  |   |  |   |
           0---1  1---0  2---3  3---2

    .. code-block:: python3

        import ufl
        import FIAT
        import finat

        cell = FIAT.ufc_cell(ufl.interval)
        elem = finat.DiscontinuousLagrange(cell, 1)
        elem = finat.TensorProductElement([elem, elem])
        print(elem.entity_permutations)

    prints:

    {(0, 0): {0: {(0, 0, 0): []},
              1: {(0, 0, 0): []},
              2: {(0, 0, 0): []},
              3: {(0, 0, 0): []}},
     (0, 1): {0: {(0, 0, 0): [],
                  (0, 0, 1): []},
              1: {(0, 0, 0): [],
                  (0, 0, 1): []}},
     (1, 0): {0: {(0, 0, 0): [],
                  (0, 1, 0): []},
              1: {(0, 0, 0): [],
                  (0, 1, 0): []}},
     (1, 1): {0: {(0, 0, 0): [0, 1, 2, 3],
                  (0, 0, 1): [1, 0, 3, 2],
                  (0, 1, 0): [2, 3, 0, 1],
                  (0, 1, 1): [3, 2, 1, 0],
                  (1, 0, 0): [0, 2, 1, 3],
                  (1, 0, 1): [2, 0, 3, 1],
                  (1, 1, 0): [1, 3, 0, 2],
                  (1, 1, 1): [3, 1, 2, 0]}}}
    """
    nfactors = len(factors)
    permutations = {}
    for dim in product(*[fe.cell.get_topology().keys() for fe in factors]):
        dim_permutations = []
        e_o_p_maps = [fe.entity_permutations[d] for fe, d in zip(factors, dim)]
        for e_tuple in product(*[sorted(e_o_p_map) for e_o_p_map in e_o_p_maps]):
            # Handle extrinsic orientations.
            # This is complex and we need to think to make this function more general.
            # One interesting case is pyramid x pyramid. There are two types of facets
            # in a pyramid cell, quad and triangle, and two types of intervals, ones
            # attached to quad (Iq) and ones attached to triangles (It). When we take
            # a tensor product of two pyramid cells, there are different kinds of tensor
            # product of intervals, i.e., Iq x Iq, Iq x It, It x Iq, It x It, and we
            # need a careful thought on how many possible extrinsic orientations we need
            # to consider for each.
            # For now we restrict ourselves to specific cases.
            cells = [fe.cell for fe in factors]
            if len(set(cells)) == len(cells):
                # All components have different cells.
                # Example: triangle x interval.
                #          dim == (2, 1) ->
                #          triangle x interval (1 possible extrinsic orientation).
                axis_perms = (tuple(range(len(factors))), )  # Identity: no permutations
            elif len(set(cells)) == 1 and isinstance(cells[0], FIAT.reference_element.UFCInterval):
                # Tensor product of intervals.
                # Example: interval x interval x interval x interval
                #          dim == (0, 1, 1, 1) ->
                #          point x interval x interval x interval  (1! * 3! possible extrinsic orientations).
                axis_perms = sorted(itertools.permutations(range(len(factors))))
                for idim, d in enumerate(dim):
                    if d == 0:
                        # idim-th component does not contribute to the extrinsic orientation.
                        axis_perms = [ap for ap in axis_perms if ap[idim] == idim]
            else:
                # More general tensor product cells.
                # Example: triangle x quad x triangle x triangle x interval x interval
                #          dim == (2, 2, 2, 2, 1, 1) ->
                #          triangle x quad x triangle x triangle x interval x interval (3! * 1! * 2! possible extrinsic orientations).
                raise NotImplementedError(f"Unable to compose permutations for {' x '.join([str(fe) for fe in factors])}")
            o_tuple_perm_map = {}
            for eo, ap in enumerate(axis_perms):
                o_p_maps = [e_o_p_map[e] for e_o_p_map, e in zip(e_o_p_maps, e_tuple)]
                for o_tuple in product(*[o_p_map.keys() for o_p_map in o_p_maps]):
                    ps = [o_p_map[o] for o_p_map, o in zip(o_p_maps, o_tuple)]
                    shape = [len(p) for p in ps]
                    for idim in range(len(ap)):
                        shape[ap[idim]] = len(ps[idim])
                    size = numpy.prod(shape)
                    if size == 0:
                        o_tuple_perm_map[(eo, ) + o_tuple] = []
                    else:
                        a = numpy.arange(size).reshape(shape)
                        # Tensorproduct elements on a tensorproduct cell of intervals:
                        # When we map the reference element to the physical element, we fisrt apply
                        # the extrinsic orientation and then the intrinsic orientation.
                        # Thus, to make the a.reshape(-1) trick work in the below,
                        # we apply the inverse operation on a; we first apply the inverse of the
                        # intrinsic orientation and then the inverse of the extrinsic orienataion.
                        for idim, p in enumerate(ps):
                            # Note that p inverse = p for interval elements.
                            # Do not use p inverse (just use p) for elements on simplices
                            # as p already does what we want by construction.
                            a = a.swapaxes(0, ap[idim])[p, :].swapaxes(0, ap[idim])
                        apinv = list(range(nfactors))
                        for idim in range(len(ap)):
                            apinv[ap[idim]] = idim
                        a = numpy.moveaxis(a, range(nfactors), apinv)
                        o_tuple_perm_map[(eo, ) + o_tuple] = a.reshape(-1).tolist()
            dim_permutations.append((e_tuple, o_tuple_perm_map))
        permutations[dim] = dict(enumerate(v for k, v in sorted(dim_permutations)))
    return permutations


def factor_point_set(product_cell, product_dim, point_set):
    """Factors a point set for the product element into a point sets for
    each subelement.

    :arg product_cell: a TensorProductCell
    :arg product_dim: entity dimension for the product cell
    :arg point_set: point set for the product element
    """
    assert len(product_cell.cells) == len(product_dim)
    point_dims = [cell.construct_subelement(dim).get_spatial_dimension()
                  for cell, dim in zip(product_cell.cells, product_dim)]

    if isinstance(point_set, TensorPointSet) and \
       len(product_cell.cells) == len(point_set.factors):
        # Just give the factors asserting matching dimensions.
        assert len(point_set.factors) == len(point_dims)
        assert all(ps.dimension == dim
                   for ps, dim in zip(point_set.factors, point_dims))
        return point_set.factors

    # Split the point coordinates along the point dimensions
    # required by the subelements.
    assert point_set.dimension == sum(point_dims)
    slices = TensorProductCell._split_slices(point_dims)
    if isinstance(point_set, PointSingleton):
        return [PointSingleton(point_set.point[s]) for s in slices]
    elif isinstance(point_set, (PointSet, TensorPointSet)):
        # Use the same point index for the new point sets.
        result = []
        for s in slices:
            ps = PointSet(point_set.points[:, s])
            ps.indices = point_set.indices
            result.append(ps)
        return result

    raise NotImplementedError("How to tabulate TensorProductElement on %s?" % (type(point_set).__name__,))
