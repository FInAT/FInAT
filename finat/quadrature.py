from abc import ABCMeta, abstractproperty
from functools import reduce

import numpy

import gem
from gem.utils import cached_property

from FIAT.reference_element import LINE, QUADRILATERAL, TENSORPRODUCT
from FIAT.quadrature import (GaussLegendreQuadratureLineRule,
                             GaussLobattoLegendreQuadratureLineRule)
from FIAT.quadrature_schemes import create_quadrature as fiat_scheme

from finat.point_set import (GaussLegendrePointSet, GaussLobattoLegendrePointSet,
                             PointSet, TensorPointSet)


def make_quadrature(ref_el, degree, scheme="default"):
    """
    Generate quadrature rule for given reference element
    that will integrate an polynomial of order 'degree' exactly.

    For low-degree polynomials on triangles (<=25) and tetrahedra (<=15), this
    uses hard-coded rules from Xiao and Gimbutas, otherwise it falls back to a
    collapsed Gauss scheme on simplices.  On tensor-product cells, it is a
    tensor-product Gauss-Lobatto quadrature rule of the subcells.

    :arg ref_el: The FIAT cell to create the quadrature for.
    :arg degree: The degree of polynomial that the rule should integrate exactly.
    :arg scheme: The quadrature scheme. Choose from
        'default' FIAT chooses the scheme as described above,
        'canonical' for the collapsed Gauss scheme,
        'GLL' for the Gauss-Lobatto-Legendre scheme (tensor-product cells only),
            degree=2*k-1 gives a diagonal mass matrix for CG_k,
        'KMV' for the Kong-Mulder-Veldhuizen scheme (simplices only).
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

    if ref_el.get_shape() == LINE:
        if scheme.lower() == "gll":
            num_points = 1 + (degree + 1 + 1) // 2
            make_fiat_rule = GaussLobattoLegendreQuadratureLineRule
            make_point_set = GaussLobattoLegendrePointSet
        elif scheme.lower() in ["gl", "default", "canonical"]:
            # FIAT uses Gauss-Legendre line quadature, however, since we
            # symbolically label it as such, we wish not to risk attaching
            # the wrong label in case FIAT changes.  So we explicitly ask
            # for Gauss-Legendre line quadature.
            num_points = (degree + 1 + 1) // 2
            make_fiat_rule = GaussLegendreQuadratureLineRule
            make_point_set = GaussLegendrePointSet
        else:
            raise ValueError(f"Invalid quadrature scheme {scheme}.")

        fiat_rule = make_fiat_rule(ref_el, num_points)
        point_set = make_point_set(fiat_rule.get_points())
        return QuadratureRule(point_set, fiat_rule.get_weights())

    fiat_rule = fiat_scheme(ref_el, degree, scheme)
    return QuadratureRule(PointSet(fiat_rule.get_points()), fiat_rule.get_weights())


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


class QuadratureRule(AbstractQuadratureRule):
    """Generic quadrature rule with no internal structure."""

    def __init__(self, point_set, weights):
        weights = numpy.asarray(weights)
        assert len(point_set.points) == len(weights)

        self.point_set = point_set
        self.weights = numpy.asarray(weights)

    @cached_property
    def point_set(self):
        pass  # set at initialisation

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
