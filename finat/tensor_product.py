from __future__ import absolute_import, print_function, division
from six.moves import range, zip

from functools import reduce
from itertools import chain, product

import numpy

from FIAT.polynomial_set import mis
from FIAT.reference_element import TensorProductCell

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.point_set import PointSet, TensorPointSet


class TensorProductElement(FiniteElementBase):

    def __init__(self, factors):
        super(TensorProductElement, self).__init__()
        self.factors = tuple(factors)
        assert all(fe.value_shape == () for fe in self.factors)

    @cached_property
    def cell(self):
        return TensorProductCell(*[fe.cell for fe in self.factors])

    @property
    def degree(self):
        return tuple(fe.degree for fe in self.factors)

    @cached_property
    def _entity_dofs(self):
        shape = tuple(fe.space_dimension() for fe in self.factors)
        entity_dofs = {}
        for dim in product(*[fe.cell.get_topology().keys()
                             for fe in self.factors]):
            entity_dofs[dim] = {}
            topds = [fe.entity_dofs()[d]
                     for fe, d in zip(self.factors, dim)]
            for tuple_ei in product(*[sorted(topd) for topd in topds]):
                tuple_vs = list(product(*[topd[ei]
                                          for topd, ei in zip(topds, tuple_ei)]))
                if tuple_vs:
                    vs = list(numpy.ravel_multi_index(numpy.transpose(tuple_vs), shape))
                else:
                    vs = []
                entity_dofs[dim][tuple_ei] = vs
            # flatten entity numbers
            entity_dofs[dim] = dict(enumerate(entity_dofs[dim][key]
                                              for key in sorted(entity_dofs[dim])))
        return entity_dofs

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return numpy.prod([fe.space_dimension() for fe in self.factors])

    @property
    def index_shape(self):
        return tuple(chain(*[fe.index_shape for fe in self.factors]))

    @property
    def value_shape(self):
        return ()  # TODO: non-scalar factors not supported yet

    def basis_evaluation(self, order, ps, entity=None):
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

        # Factor point set
        ps_factors = factor_point_set(self.cell, entity_dim, ps)

        # Subelement results
        factor_results = [fe.basis_evaluation(order, ps_, e)
                          for fe, ps_, e in zip(self.factors, ps_factors, entities)]

        # Spatial dimension
        dimension = self.cell.get_spatial_dimension()

        # A list of slices that are used to select dimensions
        # corresponding to each subelement.
        dim_slices = TensorProductCell._split_slices([c.get_spatial_dimension()
                                                      for c in self.cell.cells])

        # A list of multiindices, one multiindex per subelement, each
        # multiindex describing the shape of basis functions of the
        # subelement.
        alphas = [fe.get_indices() for fe in self.factors]

        result = {}
        for derivative in range(order + 1):
            for Delta in mis(dimension, derivative):
                # Split the multiindex for the subelements
                deltas = [Delta[s] for s in dim_slices]
                # GEM scalars (can have free indices) for collecting
                # the contributions from the subelements.
                scalars = []
                for fr, delta, alpha in zip(factor_results, deltas, alphas):
                    # Turn basis shape to free indices, select the
                    # right derivative entry, and collect the result.
                    scalars.append(gem.Indexed(fr[delta], alpha))
                # Multiply the values from the subelements and wrap up
                # non-point indices into shape.
                result[Delta] = gem.ComponentTensor(
                    reduce(gem.Product, scalars),
                    tuple(chain(*alphas))
                )
        return result


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
    elif isinstance(point_set, PointSet):
        # Split the point coordinates along the point dimensions
        # required by the subelements, but use the same point index
        # for the new point sets.
        assert point_set.dimension == sum(point_dims)
        slices = TensorProductCell._split_slices(point_dims)
        result = []
        for s in slices:
            ps = PointSet(point_set.points[:, s])
            ps.indices = point_set.indices
            result.append(ps)
        return result
    else:
        raise NotImplementedError("How to tabulate TensorProductElement on %s?" % (type(point_set).__name__,))
