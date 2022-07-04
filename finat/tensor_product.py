from functools import reduce
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
    """
    permutations = {}
    for dim in product(*[fe.cell.get_topology().keys()
                         for fe in factors]):
        dim_permutations = []
        e_o_p_maps = [fe.entity_permutations[d]
                      for fe, d in zip(factors, dim)]
        for e_tuple in product(*[sorted(e_o_p_map) for e_o_p_map in e_o_p_maps]):
            o_p_maps = [e_o_p_map[e] for e_o_p_map, e in zip(e_o_p_maps, e_tuple)]
            o_tuple_perm_map = {}
            for o_tuple in product(*[o_p_map.keys() for o_p_map in o_p_maps]):
                ps = [o_p_map[o] for o_p_map, o in zip(o_p_maps, o_tuple)]
                shape = tuple(len(p) for p in ps)
                size = numpy.prod(shape)
                if size == 0:
                    o_tuple_perm_map[o_tuple] = []
                else:
                    a = numpy.arange(size).reshape(shape)
                    for i, p in enumerate(ps):
                        a = a.swapaxes(0, i)[p, :].swapaxes(0, i)
                    o_tuple_perm_map[o_tuple] = a.reshape(-1).tolist()
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

    if isinstance(point_set, TensorPointSet):
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
    elif isinstance(point_set, PointSet):
        # Use the same point index for the new point sets.
        result = []
        for s in slices:
            ps = PointSet(point_set.points[:, s])
            ps.indices = point_set.indices
            result.append(ps)
        return result

    raise NotImplementedError("How to tabulate TensorProductElement on %s?" % (type(point_set).__name__,))
