from __future__ import absolute_import, print_function, division
from six.moves import range, zip

from functools import reduce
from itertools import chain

import numpy

from FIAT.reference_element import TensorProductCell

import gem

from finat.finiteelementbase import FiniteElementBase
from finat.point_set import TensorPointSet


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

    def basis_evaluation(self, ps, entity=None, derivative=0):
        if not isinstance(ps, TensorPointSet):
            raise NotImplementedError("How to tabulate TensorProductElement on non-TensorPointSet?")
        assert len(ps.factors) == len(self.factors)

        if entity is None:
            entity = (self.cell.get_dimension(), 0)
        entity_dim, entity_id = entity

        assert isinstance(entity_dim, tuple)
        assert len(entity_dim) == len(self.factors)

        shape = tuple(len(c.get_topology()[d])
                      for c, d in zip(self.cell.cells, entity_dim))
        entities = list(zip(entity_dim, numpy.unravel_index(entity_id, shape)))

        dimension = self.cell.get_spatial_dimension()
        tensor = numpy.empty((dimension,) * derivative, dtype=object)
        eye = numpy.eye(dimension, dtype=int)
        alphas = [fe.get_indices() for fe in self.factors]
        dim_slices = TensorProductCell._split_slices([c.get_spatial_dimension()
                                                      for c in self.cell.cells])
        for delta in numpy.ndindex(tensor.shape):
            D_ = tuple(eye[delta, :].sum(axis=0))
            Ds = [D_[s] for s in dim_slices]
            scalars = []
            for fe, ps_, e, D, alpha in zip(self.factors, ps.factors, entities, Ds, alphas):
                value = fe.basis_evaluation(ps_, entity=e, derivative=sum(D))
                d = tuple(chain(*[(dim,) * count for dim, count in enumerate(D)]))
                scalars.append(gem.Indexed(value, alpha + d))
            tensor[delta] = reduce(gem.Product, scalars)

        delta = tuple(gem.Index(extent=dimension) for i in range(derivative))
        if derivative:
            value = gem.Indexed(gem.ListTensor(tensor), delta)
        else:
            value = tensor[()]

        return gem.ComponentTensor(value, tuple(chain(*alphas)) + delta)
