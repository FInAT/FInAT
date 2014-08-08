import numpy
import pymbolic.primitives as p


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

    def kernel_variable(self, name, kernel_data):
        '''Produce a variable in the kernel data for this point set.'''
        static_key = (id(self), )

        static_data = kernel_data.static

        if static_key in static_data:
            w = static_data[static_key][0]
        else:
            w = p.Variable(name)
            data = self._points
            static_data[static_key] = (w, lambda: data)

        return w

    def __getitem__(self, i):
        if isinstance(i, int):
            return PointSet([self.points[i]])
        else:
            return PointSet(self.points[i])
