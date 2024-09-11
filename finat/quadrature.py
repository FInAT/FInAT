from abc import ABCMeta, abstractproperty
from functools import reduce

import gem
import numpy
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from FIAT.quadrature_schemes import create_quadrature as fiat_scheme
from FIAT.reference_element import LINE, QUADRILATERAL, TENSORPRODUCT
from gem.utils import cached_property

from finat.point_set import GaussLegendrePointSet, PointSet, TensorPointSet


def make_quadrature(ref_el, degree, scheme="default"):
    """
    Generate quadrature rule for given reference element
    that will integrate an polynomial of order 'degree' exactly.

    For low-degree (<=6) polynomials on triangles and tetrahedra, this
    uses hard-coded rules, otherwise it falls back to a collapsed
    Gauss scheme on simplices.  On tensor-product cells, it is a
    tensor-product quadrature rule of the subcells.

    :arg ref_el: The FIAT cell to create the quadrature for.
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
        return TensorProductQuadratureRule(quad_rules, ref_el=ref_el)

    if ref_el.get_shape() == QUADRILATERAL:
        return make_quadrature(ref_el.product, degree, scheme)

    if degree < 0:
        raise ValueError("Need positive degree, not %d" % degree)

    if ref_el.get_shape() == LINE and not ref_el.is_macrocell():
        # FIAT uses Gauss-Legendre line quadature, however, since we
        # symbolically label it as such, we wish not to risk attaching
        # the wrong label in case FIAT changes.  So we explicitly ask
        # for Gauss-Legendre line quadature.
        num_points = (degree + 1 + 1) // 2  # exact integration
        fiat_rule = GaussLegendreQuadratureLineRule(ref_el, num_points)
        point_set = GaussLegendrePointSet(fiat_rule.get_points())
        return QuadratureRule(point_set, fiat_rule.get_weights(), ref_el=ref_el, io_ornt_map_tuple=fiat_rule._intrinsic_orientation_permutation_map_tuple)

    fiat_rule = fiat_scheme(ref_el, degree, scheme)
    return QuadratureRule(PointSet(fiat_rule.get_points()), fiat_rule.get_weights(), ref_el=ref_el, io_ornt_map_tuple=fiat_rule._intrinsic_orientation_permutation_map_tuple)


class AbstractQuadratureRule(metaclass=ABCMeta):
    """Abstract class representing a quadrature rule as point set and a
    corresponding set of weights."""

    @abstractproperty
    def point_set(self):
        """Point set object representing the quadrature points."""

    @abstractproperty
    def weight_expression(self):
        """GEM expression describing the weights, with the same free indices
        as the point set."""

    @cached_property
    def extrinsic_orientation_permutation_map(self):
        """A map from extrinsic orientations to corresponding axis permutation matrices.

        Notes
        -----
        result[eo] gives the physical axis-reference axis permutation matrix corresponding to
        eo (extrinsic orientation).

        """
        if self.ref_el is None:
            raise ValueError("Must set ref_el")
        return self.ref_el.extrinsic_orientation_permutation_map

    @cached_property
    def intrinsic_orientation_permutation_map_tuple(self):
        """A tuple of maps from intrinsic orientations to corresponding point permutations for each reference cell axis.

        Notes
        -----
        result[axis][io] gives the physical point-reference point permutation array corresponding to
        io (intrinsic orientation) on ``axis``.

        """
        if any(m is None for m in self._intrinsic_orientation_permutation_map_tuple):
            raise ValueError("Must set _intrinsic_orientation_permutation_map_tuple")
        return self._intrinsic_orientation_permutation_map_tuple


class QuadratureRule(AbstractQuadratureRule):
    """Generic quadrature rule with no internal structure."""

    def __init__(self, point_set, weights, ref_el=None, io_ornt_map_tuple=(None, )):
        weights = numpy.asarray(weights)
        assert len(point_set.points) == len(weights)

        self.ref_el = ref_el
        self.point_set = point_set
        self.weights = numpy.asarray(weights)
        self._intrinsic_orientation_permutation_map_tuple = io_ornt_map_tuple

    @cached_property
    def point_set(self):
        pass  # set at initialisation

    @cached_property
    def weight_expression(self):
        return gem.Indexed(gem.Literal(self.weights), self.point_set.indices)


class TensorProductQuadratureRule(AbstractQuadratureRule):
    """Quadrature rule which is a tensor product of other rules."""

    def __init__(self, factors, ref_el=None):
        self.ref_el = ref_el
        self.factors = tuple(factors)
        self._intrinsic_orientation_permutation_map_tuple = tuple(
            m
            for factor in factors
            for m in factor._intrinsic_orientation_permutation_map_tuple
        )

    @cached_property
    def point_set(self):
        return TensorPointSet(q.point_set for q in self.factors)

    @cached_property
    def weight_expression(self):
        return reduce(gem.Product, (q.weight_expression for q in self.factors))
