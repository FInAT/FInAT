import numpy as np
from gauss_jacobi import gauss_jacobi_rule
from points import StroudPointSet, PointSet


class QuadratureRule(object):
    """Object representing a quadrature rule as a set of points
    and a set of weights.

    :param cell: The :class:`~FIAT.reference_element.ReferenceElement`
    on which this quadrature rule is defined.
    :param points: An instance of a subclass of :class:`points.PointSetBase`
    giving the points for this quadrature rule.
    :param weights: The quadrature weights. If ``points`` is a
    :class:`points.TensorPointSet` then weights is an iterable whose
    members are the sets of weights along the respective dimensions.
    """

    def __init__(self, cell, points, weights):
        self.cell = cell
        self.points = points
        self.weights = weights


class StroudQuadrature(QuadratureRule):
    def __init__(self, cell, degree):
        """Stroud quadrature rule on simplices."""

        sd = cell.get_spatial_dimension()

        if sd + 1 != len(cell.vertices):
            raise ValueError("cell must be a simplex")

        points = np.zeros((sd, degree))
        weights = np.zeros((sd, degree))

        for d in range(1, sd + 1):
            [x, w] = gauss_jacobi_rule(sd - d, 0, degree)
            points[d - 1, :] = 0.5 * (x + 1)
            weights[d - 1, :] = w

        scale = 0.5
        for d in range(1, sd + 1):
            weights[sd - d, :] *= scale
            scale *= 0.5

        super(StroudQuadrature, self).__init__(
            cell,
            StroudPointSet(map(PointSet, points)),
            weights)
