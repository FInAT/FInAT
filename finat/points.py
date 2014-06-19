import numpy


class PointSetBase(object):
    """A way of specifying a known set of points, perhaps with some
    (tensor) structure."""

    def __init__(self):
        pass

    @property
    def points(self):
        """Return a flattened numpy array of points.

        The array has shape (num_points, topological_dim).
        """

        raise NotImplementedError


class PointSet(PointSetBase):
    """A basic pointset with no internal structure."""

    def __init__(self, points):
        self._points = numpy.array(points)

    @property
    def points(self):
        """Return a flattened numpy array of points.

        The array has shape (num_points, topological_dim).
        """

        return self._points

    def __getitem__(self, i):
        if isinstance(i, int):
            return PointSet([self.points[i]])
        else:
            return PointSet(self.points[i])
