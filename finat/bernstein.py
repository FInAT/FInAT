from finiteelementbase import FiniteElementBase
from points import StroudPointSet
from ast import ForAll, Recipe, Wave, Let, IndexSum
import pymbolic.primitives as p
from indices import BasisFunctionIndex, PointIndex


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
        xi = [self._points_variable(f.points, kernel_data)
              for f in q.factors]

        qs = q.factors

        # 1D first
        if self.cell.get_spatial_dimension() == 1:
            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")
            alpha = BasisFunctionIndex(self.degree+1)
            s = 1 - xi[0][qs[0]]
            expr = Let(((r, xi[0][qs[0]]/s)),
                       IndexSum((alpha,),
                                Wave(w,
                                     alpha,
                                     s**self.degree,
                                     w * r * (self.degree-alpha)/(alpha+1.0),
                                     w * field_var[alpha])
                                )
                       )
            return Recipe(((), (), (q,)), expr)
        elif self.cell.get_spatial_dimension() == 2:
            deg = self.degree
            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")
            tmp = kernel_data.new_variable("tmp")
            alpha1 = BasisFunctionIndex(deg+1)
            alpha2 = BasisFunctionIndex(deg+1-alpha1)
            q2 = PointIndex(q.points.factor_sets[1])
            s = 1 - xi[1][q2]
            tmp_expr = Let(((r, xi[1][q2]/s),),
                           IndexSum((alpha2,),
                                    Wave(w,
                                         alpha2,
                                         s**(deg - alpha1),
                                         w * r * (deg-alpha1-alpha2)/(1.0 + alpha2),
                                         w * field_var[alpha1*(2*deg-alpha1+3)/2])
                                    )
                           )
            s = 1 - xi[0][qs[0]]
            expr = Let(((tmp, tmp_expr),
                        (r, xi[0][qs[0]]/s)),
                       IndexSum((alpha1,),
                                Wave(w,
                                     alpha1,
                                     s**deg,
                                     w * r * (deg-alpha1)/(1. + alpha1),
                                     w * tmp[alpha1, qs[1]]
                                     )
                                )
                       )

            return Recipe(((), (), (q,)), expr)
        elif self.cell.get_spatial_dimension() == 3:
            deg = self.degree
            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")
            tmp0 = kernel_data.new_variable("tmp0")
            tmp1 = kernel_data.new_variable("tmp1")
            alpha1 = BasisFunctionIndex(deg+1)
            alpha2 = BasisFunctionIndex(deg+1-alpha1)
            alpha3 = BasisFunctionIndex(deg+1-alpha1-alpha2)
            q2 = PointIndex(q.points.factor_sets[1])
            q3 = PointIndex(q.points.factor_sets[2])

            def pd(sd, d):
                if sd == 3:
                    return (d+1)*(d+2)*(d+3)/6
                elif sd == 2:
                    return (d+1)*(d+2)/2
                else:
                    raise NotImplementedError

            s = 1.0 - xi[2][q3]
            tmp0_expr = Let(((r, xi[2][q3]/s),),
                            IndexSum((alpha3,),
                                     Wave(w,
                                          alpha3,
                                          s**(deg-alpha1-alpha2),
                                          w * r * (deg-alpha1-alpha2-alpha3)/(1.+alpha3),
                                          w * field_var[pd(3, deg)-pd(3, deg-alpha1)
                                                        + pd(2, deg - alpha1)-pd(2, deg - alpha1 - alpha2)
                                                        + alpha3]
                                          )
                                     )
                            )
            s = 1.0 - xi[1][q2]
            tmp1_expr = Let(((tmp0, tmp0_expr),
                             (r, xi[1][q2]/s)),
                            IndexSum((alpha2,),
                                     Wave(w,
                                          alpha2,
                                          s**(deg-alpha1),
                                          w*r*(deg-alpha1-alpha2)/(1.0+alpha2),
                                          w*tmp0[alpha1, alpha2, q3]
                                          )
                                     )
                            )

            s = 1.0 - xi[0][qs[0]]
            expr = Let(((tmp1, tmp1_expr),
                        (r, xi[0][qs[0]]/s)),
                       IndexSum((alpha1,),
                                Wave(w,
                                     alpha1,
                                     s**deg,
                                     w*r*(deg-alpha1)/(1.+alpha1),
                                     w*tmp1[alpha1, qs[1], qs[2]]
                                     )
                                )
                       )

            return Recipe(((), (), (q,)), expr)
