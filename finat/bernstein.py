from finiteelementbase import FiniteElementBase
from points import StroudPointSet
from ast import ForAll, Recipe, Wave, Let, IndexSum
import pymbolic as p
from index import BasisFunctionIndex


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
            raise ValueError("Only Stroud points may be employed with Bernstein polynomials")

        if derivative is not None:
            raise NotImplementedError

        # Get the symbolic names for the points.
        xi = [self._points_variable(f, kernel_data)
              for f in q.factors.points]

        qs = q.factors
        r = kernel_data.new_variable("r")
        w = kernel_data.new_variable("w")
        alpha = BasisFunctionIndex(self.degree + 1)
        s = 1 - xi[qs[0]]

        # 1D first
        expr = Let(((r, xi[qs[0]] / s),),
                   IndexSum((alpha,),
                            Wave(w,
                                 alpha,
                                 s ** self.degree,
                                 w * r * (self.degree - alpha) / (alpha + 1.0),
                                 w * field_var[alpha])
                            )
                   )

        return Recipe(((), (), (q)), expr)
