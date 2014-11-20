from finiteelementbase import FiniteElementBase
from points import StroudPointSet
from ast import ForAll, Recipe, Wave, Let, IndexSum
import pymbolic.primitives as p
from indices import BasisFunctionIndex, PointIndex
import numpy as np


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

    @property
    def dofs_shape(self):

        degree = self.degree
        dim = self.cell.get_spatial_dimension()
        return (int(np.prod(xrange(degree + 1, degree + 1 + dim))
                    / np.prod(xrange(1, dim + 1))),)

    def field_evaluation(self, field_var, q, kernel_data, derivative=None):
        if not isinstance(q.points, StroudPointSet):
            raise ValueError("Only Stroud points may be employed with Bernstein polynomials")

        if derivative is not None:
            raise NotImplementedError

        def pd(sd, d):
            if sd == 3:
                return (d+1)*(d+2)*(d+3)/6
            elif sd == 2:
                return (d+1)*(d+2)/2
            elif sd == 1:
                return d+1
            else:
                raise NotImplementedError


        # Get the symbolic names for the points.
        xi = [self._points_variable(f.points, kernel_data)
              for f in q.factors]

        qs = q.factors

        deg = self.degree

        sd = self.cell.get_spatial_dimension()

        # reimplement using reduce to avoid problem with infinite loop into pymbolic
        def mysum(vals):
            return reduce(lambda a, b: a+b, vals, 0)

        # Create basis function indices that run over
        # the possible multiindex space.  These have
        # to be jagged
        alphas = [BasisFunctionIndex(deg+1)]
        for d in range(1, sd):
            asum = mysum(alphas)
            alpha_cur = BasisFunctionIndex(deg+1-asum)
            alphas.append(alpha_cur)

        qs_internal = [PointIndex(qf) for qf in q.points.factor_sets]

        r = kernel_data.new_variable("r")
        w = kernel_data.new_variable("w")
        tmps = [kernel_data.new_variable("tmp") for d in range(sd-1)]

        # every phase of sum-factorization *except* the last one
        # will make use of the qs_internal to index into
        # quadrature points, but the last phase will use the
        # index set from the point set given as an argument.
        # To avoid some if statements, we will create a list of
        # what quadrature point indices should be used at which phase
        qs_per_phase = [qs_internal]*(sd-1) + [qs]

        # The first phase is different since we read from field_var
        # instead of one of the temporaries -- field_var is stored
        # by a single index and the rest are stored as tensor products.
        # This code computes the offset into the field_var storage
        # for the internal sum variable loop.
        offset = 0
        for d in range(sd-1):
            deg_begin = deg - mysum(alphas[:d])
            deg_end = deg - alphas[d]
            offset += pd(sd-d, deg_begin) - pd(sd-d, deg_end)

        # each phase of the sum-factored algorithm reads from a particular
        # location.  The first of these is field_var, the rest are the
        # temporaries.
        read_locs = [field_var[alphas[-1]+offset]] \
            + [tmps[d][tuple(alphas[:(-d)]+qs_internal[(-d):])]
               for d in range(sd-1)]

        # In the first phase of the sum-factorization we don't have a previous
        # result to bind, so the Let expression is different.
        qs_cur = qs_per_phase[0]
        s = 1.0 - xi[-1][qs_cur[-1]]
        expr = Let(((r, xi[-1][qs_cur[-1]]/s),),
                   IndexSum((alphas[-1],),
                            Wave(w,
                                 alphas[-1],
                                 s**(deg-mysum(alphas[:(sd-1)])),
                                 w * r * (deg-mysum(alphas)) / (1.0 + alphas[-1]),
                                 w * read_locs[0]
                                 )
                            )
                   )

        for d in range(sd-1):
            qs_cur = qs_per_phase[d+1]
            b_ind = -(d+2)  # index into several things counted backward
            s = 1.0 - xi[b_ind][qs_cur[b_ind]]

            recipe_args = ((),
                           tuple(alphas[:(b_ind+1)]),
                           tuple(qs_cur[(b_ind+1):]))

            expr = Let(((tmps[d], Recipe(recipe_args, expr)),
                        (r, xi[b_ind][qs_cur[b_ind]]/s)),
                       IndexSum((alphas[b_ind],),
                                Wave(w,
                                     alphas[b_ind],
                                     s**(deg-mysum(alphas[:b_ind])),
                                     w * r * (deg-mysum(alphas[:(b_ind+1)]))/(1.0+alphas[b_ind]),
                                     w * read_locs[d+1]
                                     )
                                )
                       )

        return Recipe(((), (), (q,)), expr)

#        if self.cell.get_spatial_dimension() == 1:
#            r = kernel_data.new_variable("r")
#            w = kernel_data.new_variable("w")
#            alpha = BasisFunctionIndex(self.degree+1)
#            s = 1 - xi[0][qs[0]]
#            expr = Let(((r, xi[0][qs[0]]/s)),
#                       IndexSum((alpha,),
#                                Wave(w,
#                                     alpha,
#                                     s**self.degree,
#                                     w * r * (self.degree-alpha)/(alpha+1.0),
#                                     w * field_var[alpha])
#                                )
#                       )
#            return Recipe(((), (), (q,)), expr)
#        elif self.cell.get_spatial_dimension() == 2:
#            deg = self.degree
#            r = kernel_data.new_variable("r")
#            w = kernel_data.new_variable("w")
#            tmp = kernel_data.new_variable("tmp")
#            alpha1 = BasisFunctionIndex(deg+1)
#            alpha2 = BasisFunctionIndex(deg+1-alpha1)
#            q2 = PointIndex(q.points.factor_sets[1])
#            s = 1 - xi[1][q2]

#            tmp_expr = Let(((r, xi[1][q2]/s),),
#                           IndexSum((alpha2,),
#                                    Wave(w,
#                                         alpha2,
#                                         s**(deg - alpha1),
#                                         w * r * (deg-alpha1-alpha2)/(1.0 + alpha2),
#                                         w * field_var[alpha1*(2*deg-alpha1+3)/2])
#                                    )
#                           )
#            s = 1 - xi[0][qs[0]]
#            expr = Let(((tmp, Recipe(((), (alpha1,), (q2,)), tmp_expr)),
#                        (r, xi[0][qs[0]]/s)),
#                       IndexSum((alpha1,),
#                                Wave(w,
#                                     alpha1,
#                                     s**deg,
#                                     w * r * (deg-alpha1)/(1. + alpha1),
#                                     w * tmp[alpha1, qs[1]]
#                                     )
#                                )
#                       )
#
#            return Recipe(((), (), (q,)), expr)
#        elif self.cell.get_spatial_dimension() == 3:
#            deg = self.degree
#            r = kernel_data.new_variable("r")
#            w = kernel_data.new_variable("w")
#            tmp0 = kernel_data.new_variable("tmp0")
#            tmp1 = kernel_data.new_variable("tmp1")
#            alpha1 = BasisFunctionIndex(deg+1)
#            alpha2 = BasisFunctionIndex(deg+1-alpha1)
#            alpha3 = BasisFunctionIndex(deg+1-alpha1-alpha2)
#            q2 = PointIndex(q.points.factor_sets[1])
#            q3 = PointIndex(q.points.factor_sets[2])
#
#            s = 1.0 - xi[2][q3]
#            tmp0_expr = Let(((r, xi[2][q3]/s),),
#                            IndexSum((alpha3,),
#                                     Wave(w,
#                                          alpha3,
#                                          s**(deg-alpha1-alpha2),
#                                          w * r * (deg-alpha1-alpha2-alpha3)/(1.+alpha3),
#                                          w * field_var[pd(3, deg)-pd(3, deg-alpha1)
#                                                        + pd(2, deg - alpha1)-pd(2, deg - alpha1 -# alpha2)
#                                                        + alpha3]
#                                          )
#                                     )
#                            )
#            s = 1.0 - xi[1][q2]
#            tmp1_expr = Let(((tmp0, tmp0_expr),
#                             (r, xi[1][q2]/s)),
#                            IndexSum((alpha2,),
#                                     Wave(w,
#                                          alpha2,
#                                          s**(deg-alpha1),
#                                          w*r*(deg-alpha1-alpha2)/(1.0+alpha2),
#                                          w*tmp0[alpha1, alpha2, q3]
#                                          )
#                                     )
#                            )
#
#            s = 1.0 - xi[0][qs[0]]
#            expr = Let(((tmp1, tmp1_expr),
#                        (r, xi[0][qs[0]]/s)),
#                       IndexSum((alpha1,),
#                                Wave(w,
#                                     alpha1,
#                                     s**deg,
#                                     w*r*(deg-alpha1)/(1.+alpha1),
#                                     w*tmp1[alpha1, qs[1], qs[2]]
#                                     )
#                                )
#                       )
#
#            return Recipe(((), (), (q,)), expr)
