import numpy as np
from gauss_jacobi import gauss_jacobi_rule
from points import StroudPointSet, PointSet, TensorPointSet, GaussLobattoPointSet
import FIAT
import gem


def make_quadrature(cell, degree, preferred_quadrature):
    '''Return a quadrature rule of the suggested degree in accordance with
    the quadrature preferences provided.'''

    # Currently just do the dumb thing. Smart quadrature happens later.
    return CollapsedGaussJacobiQuadrature(cell, degree)


class QuadratureRule(object):
    """Object representing a quadrature rule as a set of points
    and a set of weights.

    :param cell: The :class:`~FIAT.reference_element.ReferenceElement`
    on which this quadrature rule is defined.
    """

    def __init__(self, cell, points, weights):
        self.cell = cell
        self._points = points
        self._weights = weights

    @property
    def points(self):
        '''The quadrature points. For a rule with internal structure, this is
        the flattened points.'''

        return self._points

    @property
    def weights(self):
        '''The quadrature weights. For a rule with internal structure, this is
        the flattenened weights.'''

        return self._weights

    @property
    def index_shape(self):
        '''A tuple indicating the shape of the indices needed to loop over the points.'''

        raise NotImplementedError

    def get_indices(self):
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        return tuple(gem.Index(extent=d) for d in self.index_shape)


class CollapsedGaussJacobiQuadrature(QuadratureRule):
    def __init__(self, cell, degree):
        """Gauss Jacobi Quadrature rule using collapsed coordinates to
        accommodate higher order simplices."""

        points = (degree + 1) // 2

        rule = FIAT.make_quadrature(cell, points)

        super(CollapsedGaussJacobiQuadrature, self).__init__(cell,
                                                             rule.pts,
                                                             rule.wts)

    @property
    def index_shape(self):
        '''A tuple indicating the shape of the indices needed to loop over the points.'''

        return (len(self.points),)


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


class GaussLobattoQuadrature(QuadratureRule):
    def __init__(self, cell, points):
        """Gauss-Lobatto-Legendre quadrature on hypercubes.
        :param cell: The reference cell on which to define the quadrature.
        :param points: The number of points, or a tuple giving the number of
          points in each dimension.
        """

        def expand_quad(cell, points):
            d = cell.get_spatial_dimension()
            if d == 1:
                return ((cell, points,
                         FIAT.quadrature.GaussLobattoQuadratureLineRule(cell, points[0])),)
            else:
                try:
                    d_a = cell.A.get_spatial_dimension()
                    return expand_quad(cell.A, points[:d_a])\
                        + expand_quad(cell.B, points[d_a:])
                except AttributeError():
                    raise ValueError("Unable to create Gauss-Lobatto quadrature on ",
                                     + str(cell))
        try:
            points = tuple(points)
        except TypeError:
            points = (points,)

        if len(points) == 1:
            points *= cell.get_spatial_dimension()

        cpq = expand_quad(cell, points)

        # uniquify q.
        lookup = {(c, p): (GaussLobattoPointSet(q.get_points()),
                           PointSet(q.get_weights())) for c, p, q in cpq}
        pointset = tuple(lookup[c, p][0] for c, p, _ in cpq)
        weightset = tuple(lookup[c, p][1] for c, p, _ in cpq)

        if len(cpq) == 1:
            super(GaussLobattoQuadrature, self).__init__(
                cell, pointset[0], weightset[0])
        else:
            super(GaussLobattoQuadrature, self).__init__(
                cell,
                TensorPointSet(pointset),
                weightset)
