from __future__ import absolute_import, print_function, division

from FIAT.polynomial_set import mis
from FIAT.reference_element import LINE

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase


class RuntimeTabulated(FiniteElementBase):

    def __init__(self, cell, degree, variant=None, shift_axes=0, restriction=None, continuous=True):
        if cell.get_shape() != LINE:
            raise NotImplementedError("Runtime tabulated elements limited to 1D.")

        assert isinstance(variant, str)
        assert isinstance(shift_axes, int) and 0 <= shift_axes
        assert isinstance(continuous, bool)
        assert restriction in [None, '+', '-']

        self.cell = cell
        self.degree = degree
        self.variant = variant
        self.shift_axes = shift_axes
        self.restriction = restriction
        self.continuous = continuous

    @cached_property
    def cell(self):
        pass  # set at initialization

    @cached_property
    def degree(self):
        pass  # set at initialization

    @cached_property
    def formdegree(self):
        if self.continuous:
            return 0
        else:
            return self.cell.get_spatial_dimension()

    def entity_dofs(self):
        raise NotImplementedError("I cannot tell where my DoFs are... :-/")

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
                name = str.format("rt_{}{}d{}sa{}{}{}",
                                  self.variant,
                                  self.degree,
                                  ''.join(map(str, alpha)),
                                  self.shift_axes,
                                  'c' if self.continuous else 'd',
                                  {None: "",
                                   '+': "_p",
                                   '-': "_m"}[self.restriction])
                result[alpha] = gem.partial_indexed(gem.Variable(name, shape), ps.indices)
        return result

    def point_evaluation(self, order, point, entity=None):
        raise NotImplementedError("Point evaluation supported for runtime tabulated elements")

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    @property
    def value_shape(self):
        return ()

    @property
    def mapping(self):
        return "affine"
