from __future__ import absolute_import, print_function, division
from six.moves import range, zip

from functools import reduce
from itertools import chain

import numpy

from FIAT.reference_element import TensorProductCell

import gem

from finat.finiteelementbase import FiniteElementBase
from finat.point_set import PointSet, TensorPointSet


class TensorProductElement(FiniteElementBase):

    def __init__(self, factors):
        self.factors = tuple(factors)
        assert all(fe.value_shape == () for fe in self.factors)

        self._cell = TensorProductCell(*[fe.cell for fe in self.factors])
        # self._degree = sum(fe.degree for fe in factors)  # correct?
        # self._degree = max(fe.degree for fe in factors)  # FIAT
        self._degree = None  # not used?

    @property
    def index_shape(self):
        return tuple(chain(*[fe.index_shape for fe in self.factors]))

    @property
    def value_shape(self):
        return ()  # TODO: non-scalar factors not supported yet

    def basis_evaluation(self, ps, entity=None, derivative=0):
        if entity is None:
            entity = (self.cell.get_dimension(), 0)
        entity_dim, entity_id = entity

        assert isinstance(entity_dim, tuple)
        assert len(entity_dim) == len(self.factors)

        shape = tuple(len(c.get_topology()[d])
                      for c, d in zip(self.cell.cells, entity_dim))
        entities = list(zip(entity_dim, numpy.unravel_index(entity_id, shape)))

        ps_factors = factor_point_set(self.cell, entity_dim, ps)

        # Okay, so we need to introduce some terminology before we are
        # able to explain what is going on below.  A key difficulty is
        # the necessity to use two different derivative multiindex
        # types, and convert between them back and forth.
        #
        # First, let us consider a scalar-valued function `u` in a
        # `d`-dimensional space:
        #
        #   u : R^d -> R
        #
        # Let a = (a_1, a_2, ..., a_d) be a multiindex.  You might
        # recognise this notation:
        #
        #   D^a u
        #
        # For example, a = (1, 2) means a third derivative: first
        # derivative in the x-direction, and second derivative in the
        # y-direction.  We call `a` "canonical derivative multiindex",
        # or canonical multiindex for short.
        #
        # Now we will have populate what we call a derivative tensor.
        # For example, the third derivative of `u` is a d x d x d
        # tensor:
        #
        #   \grad \grad \grad u
        #
        # A multiindex can denote an entry of this derivative tensor.
        # For example, i = (1, 0, 1) refers deriving `u` first in the
        # y-direction (second direction, indexing starts at zero),
        # then in the x-direction, and finally in the second direction
        # again.  Usually the order of these derivatives does not
        # matter, so the derivative tensor is symmetric.  For specific
        # conditions about when this actually holds, please refer to
        # your favourite calculus textbook.  We call `i` here a
        # "derivative tensor multiindex", or tensor multiindex for
        # short.  For example, the following tensor multiindices
        # correspond to the (1, 2) canonical multiindex:
        #
        #   (0, 1, 1)
        #   (1, 0, 1)
        #   (1, 1, 0)
        #
        # Now you should be ready to get go.

        # Spatial dimension
        dimension = self.cell.get_spatial_dimension()
        # The derivative tensor
        tensor = numpy.empty((dimension,) * derivative, dtype=object)
        # The identity matrix, used to facilitate conversion from
        # tensor multiindex to canonical multiindex form.
        eye = numpy.eye(dimension, dtype=int)
        # A list of multiindices, one multiindex per subelement, each
        # multiindex describing the shape of basis functions of the
        # subelement.
        alphas = [fe.get_indices() for fe in self.factors]
        # A list of slices that are used to select dimensions
        # corresponding to each subelement.
        dim_slices = TensorProductCell._split_slices([c.get_spatial_dimension()
                                                      for c in self.cell.cells])
        # 'delta' is a tensor multiindex consisting of only fixed
        # indices for populating the entries of the derivative tensor.
        for delta in numpy.ndindex(tensor.shape):
            # Get the canonical multiindex corresponding to 'delta'.
            D_ = tuple(eye[delta, :].sum(axis=0))
            # Split this canonical multiindex for the subelements.
            Ds = [D_[s] for s in dim_slices]
            # GEM scalars (can have free indices) for collecting the
            # contributions from the subelements.
            scalars = []
            for fe, ps_, e, D, alpha in zip(self.factors, ps_factors, entities, Ds, alphas):
                # Ask the subelement to tabulate at the required derivative order.
                value = fe.basis_evaluation(ps_, entity=e, derivative=sum(D))
                # Nice, but now we have got a subelement derivative
                # tensor of the given order, while we only need a
                # specific derivative.  So we convert the subelement
                # canonical multiindex to tensor multiindex.
                d = tuple(chain(*[(dim,) * count for dim, count in enumerate(D)]))
                # Turn basis shape to free indices, select the right
                # derivative entry, and collect the result.
                scalars.append(gem.Indexed(value, alpha + d))
            # Multiply the values from the subelements and insert the
            # result into the derivative tensor.
            tensor[delta] = reduce(gem.Product, scalars)

        # Convert the derivative tensor from a numpy object array to a
        # GEM expression with only free indices (and scalar shape).
        delta = tuple(gem.Index(extent=dimension) for i in range(derivative))
        if derivative:
            value = gem.Indexed(gem.ListTensor(tensor), delta)
        else:
            value = tensor[()]

        # Wrap up non-point indices into shape
        return gem.ComponentTensor(value, tuple(chain(*alphas)) + delta)


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
