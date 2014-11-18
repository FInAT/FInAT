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

        self.extent = slice(self._points.shape[0])
        """A slice which describes how to iterate over this
        :class:`PointSet`"""

    @property
    def points(self):
        """A flattened numpy array of points.

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


class TensorPointSet(PointSetBase):
    def __init__(self, factor_sets):
        super(TensorPointSet, self).__init__()

        self.factor_sets = factor_sets

    def points(self):
        def helper(loi):
            if len(loi) == 1:
                return [[x] for x in loi[0]]
            else:
                return [[x] + y for x in loi[0] for y in helper(loi[1:])]

        return numpy.array(helper([fs.points.tolist()
                                   for fs in self.factor_sets]))


class MappedMixin(object):
    def __init__(self, *args):
        super(MappedMixin, self).__init__(*args)

    def map_points(self):
        raise NotImplementedError


class DuffyMappedMixin(MappedMixin):
    def __init__(self, *args):
        super(DuffyMappedMixin, self).__init__(*args)

    def map_points(self):
        raise NotImplementedError


class StroudPointSet(TensorPointSet, DuffyMappedMixin):
    """A set of points with the structure required for Stroud quadrature."""

    def __init__(self, factor_sets):
        super(StroudPointSet, self).__init__(factor_sets)
