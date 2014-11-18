from finiteelementbase import FiniteElementBase
from points import StroudPointSet
from ast import ForAll, Recipe, Wave, Let, IndexSum
import pymbolic as p
from index import BasisFunctionIndex, PointIndex


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

        # 1D first
        if self.cell.get_spatial_dimension() == 1:
            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")
            alpha = BasisFunctionIndex(self.degree+1)
            s = 1-xi[0][qs[0]]

            expr = Let(((r, xi[0][qs[0]]/s),),
                       IndexSum((alpha,),
                                Wave(w,
                                     alpha,
                                     s**self.degree,
                                     w * r * (self.degree - alpha) / (alpha + 1.0),
                                     w * field_var[alpha])
                                )
                       )
            return Recipe(((), (), (q)), expr)
        elif self.cell.get_spatial_dimension() == 2:
            deg = self.degree
            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")
            tmp = kernel_data.new_variable("tmp")
            alpha1 = BasisFunctionIndex(deg+1)
            alpha2 = BasisFunctionIndex(deg+1-alpha1)
            q2 = PointIndex(q.points.factor_set[1])
            s = 1-xi[0][qs[0]]
            tmp_expr = Let((r, xi[1][q2]/s),
                           IndexSum((alpha2,),
                                    Wave(w,
                                         alpha2,
                                         s**(deg - alpha1),
                                         w * r * (deg-alpha1-alpha2)/(1.0 + alpha2),
                                         w * field_var[alpha1*(2*deg-alpha1+3)/2])
                                    )
                           )
            expr = Let((tmp, tmp_expr),
                       Let((r, xi[0][qs[0]]),
                           IndexSum((alpha1,),
                                    Wave(w,
                                         alpha1,
                                         s**deg,
                                         w * r * (deg-alpha1)/(1. + alpha1),
                                         w * tmp[alpha1, qs[1]]
                                         )
                                    )
                           )
                       )
            return Recipe(((), (), (q)), expr)
        elif self.cell.get_spatial_dimension() == 3:
            deg = self.degree
            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")
            tmp0 = kernel_data.new_variable("tmp0")
            tmp1 = kernel_data.new_variable("tmp1")
            alpha1 = BasisFunctionIndex(deg+1)
            alpha2 = BasisFunctionIndex(deg+1-alpha1)
            alpha3 = BasisFunctionIndex(deg+1-alpha1-alpha2)
            q3 = PointIndex(q.points.factor_set[2])

            pass
#            tmp0_expr = Let((r, xi[2][q3]/s),
#                            IndexSum((q3,),
#                                     Wave(w,
#                                          alpha3,
#                                          s**(deg-alpha1-alpha2),
#                                          w * r * (deg-alpha1-alpha2-alpha3)/(1.+alpha3),
#                                          w * field_var[]

