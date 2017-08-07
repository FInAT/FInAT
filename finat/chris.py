from __future__ import absolute_import, print_function, division

from FIAT.polynomial_set import mis
from FIAT.reference_element import LINE

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase


class Chris(FiniteElementBase):

    def __init__(self, cell, degree, shift_axes):
        assert cell.get_shape() == LINE
        self.cell = cell
        self.degree = degree
        self.shift_axes = shift_axes

    @cached_property
    def cell(self):
        pass  # set at initialization

    @cached_property
    def degree(self):
        pass  # set at initialization

    @property
    def formdegree(self):
        return 0

    def entity_dofs(self):
        raise NotImplementedError

    def space_dimension(self):
        return self.degree + 1

    def basis_evaluation(self, order, ps, entity=None):
        """Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        """
        # Spatial dimension
        dimension = self.cell.get_spatial_dimension()

        # Shape of the tabulation matrix
        shape = tuple(index.extent for index in ps.indices) + self.index_shape + self.value_shape

        result = {}
        for derivative in range(order + 1):
            for alpha in mis(dimension, derivative):
                name = "chris{}d{}sa{}".format(self.degree, ''.join(map(str, alpha)), self.shift_axes)
                result[alpha] = gem.partial_indexed(gem.Variable(name, shape), ps.indices)
        return result

    def point_evaluation(self, order, point, entity=None):
        raise NotImplementedError

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    @property
    def value_shape(self):
        return ()

    @property
    def mapping(self):
        return "affine"


class DiscontinuousChris(Chris):

    @property
    def formdegree(self):
        return self.cell.get_spatial_dimension()
