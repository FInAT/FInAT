from __future__ import absolute_import, print_function, division
from six import with_metaclass

from abc import ABCMeta, abstractproperty
from functools import reduce

import numpy

import gem
from gem.utils import cached_property

from FIAT.reference_element import QUADRILATERAL, TENSORPRODUCT
# from FIAT.quadrature import compute_gauss_jacobi_rule as gauss_jacobi_rule
from FIAT.quadrature_schemes import create_quadrature as fiat_scheme

from finat.point_set import PointSet, TensorPointSet


def make_quadrature(ref_el, degree, scheme="default"):
    """
    Generate quadrature rule for given reference element
    that will integrate an polynomial of order 'degree' exactly.

    For low-degree (<=6) polynomials on triangles and tetrahedra, this
    uses hard-coded rules, otherwise it falls back to a collapsed
    Gauss scheme on simplices.  On tensor-product cells, it is a
    tensor-product quadrature rule of the subcells.

    :arg cell: The FIAT cell to create the quadrature for.
    :arg degree: The degree of polynomial that the rule should
        integrate exactly.
    """
    if ref_el.get_shape() == TENSORPRODUCT:
        try:
            degree = tuple(degree)
        except TypeError:
            degree = (degree,) * len(ref_el.cells)

        assert len(ref_el.cells) == len(degree)
        quad_rules = [make_quadrature(c, d, scheme)
                      for c, d in zip(ref_el.cells, degree)]
        return TensorProductQuadratureRule(quad_rules)

    if ref_el.get_shape() == QUADRILATERAL:
        return make_quadrature(ref_el.product, degree, scheme)

    if degree < 0:
        raise ValueError("Need positive degree, not %d" % degree)

    fiat_rule = fiat_scheme(ref_el, degree, scheme)
    return QuadratureRule(fiat_rule.get_points(), fiat_rule.get_weights())


class AbstractQuadratureRule(with_metaclass(ABCMeta)):
    """Abstract class representing a quadrature rule as point set and a
    corresponding set of weights."""

    @abstractproperty
    def point_set(self):
        """Point set object representing the quadrature points."""

    @abstractproperty
    def weight_expression(self):
        """GEM expression describing the weights, with the same free indices
        as the point set."""


class QuadratureRule(AbstractQuadratureRule):
    """Generic quadrature rule with no internal structure."""

    def __init__(self, points, weights):
        weights = numpy.asarray(weights)
        assert len(points) == len(weights)

        self._points = numpy.asarray(points)
        self.weights = numpy.asarray(weights)

    @cached_property
    def point_set(self):
        return PointSet(self._points)

    @cached_property
    def weight_expression(self):
        return gem.Indexed(gem.Literal(self.weights), self.point_set.indices)


class TensorProductQuadratureRule(AbstractQuadratureRule):
    """Quadrature rule which is a tensor product of other rules."""

    def __init__(self, factors):
        self.factors = tuple(factors)

    @cached_property
    def point_set(self):
        return TensorPointSet(q.point_set for q in self.factors)

    @cached_property
    def weight_expression(self):
        return reduce(gem.Product, (q.weight_expression for q in self.factors))

    # def refactor(self, dims):
    #     """Refactor this quadrature rule into a tuple of quadrature rules with
    #     the dimensions specified."""

    #     qs = []
    #     i = 0
    #     next_i = 0
    #     for dim in dims:
    #         i = next_i
    #         next_dim = 0
    #         while next_dim < dim:
    #             next_dim += self.factors[next_i].spatial_dimension
    #             if next_dim > dim:
    #                 raise ValueError("Element and quadrature incompatible")
    #             next_i += 1
    #         if next_i - i > 1:
    #             qs.append(TensorProductQuadratureRule(*self.factors[i: next_i]))
    #         else:
    #             qs.append(self.factors[i])
    #     return tuple(qs)


# class StroudQuadrature(QuadratureRule):
#     def __init__(self, cell, degree):
#         """Stroud quadrature rule on simplices."""

#         sd = cell.get_spatial_dimension()

#         if sd + 1 != len(cell.vertices):
#             raise ValueError("cell must be a simplex")

#         points = numpy.zeros((sd, degree))
#         weights = numpy.zeros((sd, degree))

#         for d in range(1, sd + 1):
#             [x, w] = gauss_jacobi_rule(sd - d, 0, degree)
#             points[d - 1, :] = 0.5 * (x + 1)
#             weights[d - 1, :] = w

#         scale = 0.5
#         for d in range(1, sd + 1):
#             weights[sd - d, :] *= scale
#             scale *= 0.5

#         super(StroudQuadrature, self).__init__(
#             cell,
#             StroudPointSet(map(PointSet, points)),
#             weights)


# class GaussLobattoQuadrature(QuadratureRule):
#     def __init__(self, cell, points):
#         """Gauss-Lobatto-Legendre quadrature on hypercubes.
#         :param cell: The reference cell on which to define the quadrature.
#         :param points: The number of points, or a tuple giving the number of
#           points in each dimension.
#         """

#         def expand_quad(cell, points):
#             d = cell.get_spatial_dimension()
#             if d == 1:
#                 return ((cell, points,
#                          FIAT.quadrature.GaussLobattoQuadratureLineRule(cell, points[0])),)
#             else:
#                 try:
#                     # Note this requires generalisation for n-way products.
#                     d_a = cell.A.get_spatial_dimension()
#                     return expand_quad(cell.A, points[:d_a])\
#                         + expand_quad(cell.B, points[d_a:])
#                 except AttributeError():
#                     raise ValueError("Unable to create Gauss-Lobatto quadrature on ",
#                                      + str(cell))
#         try:
#             points = tuple(points)
#         except TypeError:
#             points = (points,)

#         if len(points) == 1:
#             points *= cell.get_spatial_dimension()

#         cpq = expand_quad(cell, points)

#         # uniquify q.
#         lookup = {(c, p): (GaussLobattoPointSet(q.get_points()),
#                            PointSet(q.get_weights())) for c, p, q in cpq}
#         pointset = tuple(lookup[c, p][0] for c, p, _ in cpq)
#         weightset = tuple(lookup[c, p][1] for c, p, _ in cpq)

#         if len(cpq) == 1:
#             super(GaussLobattoQuadrature, self).__init__(
#                 cell, pointset[0], weightset[0])
#         else:
#             super(GaussLobattoQuadrature, self).__init__(
#                 cell,
#                 TensorPointSet(pointset),
#                 weightset)
