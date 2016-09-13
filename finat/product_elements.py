"""Preliminary support for tensor product elements."""

from __future__ import absolute_import, print_function, division

from .finiteelementbase import FiniteElementBase
from FIAT.reference_element import TensorProductCell


class ProductElement(object):
    """Mixin class describing product elements."""


class ScalarProductElement(ProductElement, FiniteElementBase):
    """A scalar-valued tensor product element."""
    def __init__(self, *args):
        super(ScalarProductElement, self).__init__()

        assert all([isinstance(e, FiniteElementBase) for e in args])

        self.factors = tuple(args)

        self._degree = max([a._degree for a in args])

        cellprod = lambda cells: TensorProductCell(cells[0], cells[1] if len(cells) < 3
                                                   else cellprod(cells[1:]))

        self._cell = cellprod([a.cell for a in args])

    def __hash__(self):
        """ScalarProductElements are equal if their factors are equal"""

        return hash(self.factors)

    def __eq__(self, other):
        """VectorFiniteElements are equal if they have the same base element
        and dimension."""

        return self.factors == other.factors
