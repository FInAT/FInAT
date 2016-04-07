from .finiteelementbase import FiniteElementBase
import numpy as np


def pd(sd, d):
    if sd == 3:
        return (d + 1) * (d + 2) * (d + 3) / 6
    elif sd == 2:
        return (d + 1) * (d + 2) / 2
    elif sd == 1:
        return d + 1
    else:
        raise NotImplementedError


class Bernstein(FiniteElementBase):
    """Scalar - valued Bernstein element. Note: need to work out the
    correct heirarchy for different Bernstein elements."""

    def __init__(self, cell, degree):
        super(Bernstein, self).__init__()

        self._cell = cell
        self._degree = degree

    @property
    def dofs_shape(self):

        degree = self.degree
        dim = self.cell.get_spatial_dimension()
        return (int(np.prod(xrange(degree + 1, degree + 1 + dim)) /
                    np.prod(xrange(1, dim + 1))),)
