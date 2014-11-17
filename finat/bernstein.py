from finiteelementbase import FiniteElementBase
from points import StroudPointSet
from ast import ForAll, Recipe, Wave

class Bernstein(FiniteElementBase):
    """Scalar-valued Bernstein element. Note: need to work out the
    correct heirarchy for different Bernstein elements."""

    def __init__(self, cell, degree):
        super(Bernstein, self).__init__()

        self._cell = cell
        self._degree = degree

    def _points_variable(self, points, kernel_data):
        """Return the symbolic variables for the static data
        corresponding to the :class:`PointSet` ``points``."""

        static_key = (id(points),)
        if static_key in kernel_data.static:
            xi = kernel_data.static[static_key][0]
        else:
            xi = p.Variable(kernel_data.point_variable_name(points))
            kernel_data.static[static_key] = (xi, lambda: points.points)

        return xi

    def field_evaluation(self, field_var, q, kernel_data, derivative=None):

        if not isinstance(q.points, StroudPointSet):
            raise ValueError("Only Stroud points may be employed with Bernstien polynomials")
        
        # Get the symbolic names for the points.
        xi = [self._points_variable(f, kernel_data)
              for f in q.factors.points]

        qs = q.factors[0]

        # 1D first
        ForAll(,
               
               )
