from .finiteelementbase import FiniteElementBase
from .points import StroudPointSet
from .ast import ForAll, Recipe, Wave, Let, IndexSum, Variable
from .indices import BasisFunctionIndex, PointIndex, SimpliciallyGradedBasisFunctionIndex, DimensionIndex  # noqa
from .derivatives import grad
from .points import PointSet  # noqa

import numpy as np


# reimplement sum using reduce to avoid problem with infinite loop
# into pymbolic
def mysum(vals):
    return reduce(lambda a, b: a + b, vals, 0)


def pd(sd, d):
    if sd == 3:
        return (d + 1) * (d + 2) * (d + 3) / 6
    elif sd == 2:
        return (d + 1) * (d + 2) / 2
    elif sd == 1:
        return d + 1
    else:
        raise NotImplementedError


# these are used in pattern construction
def berntuples(sdim, deg):
    assert sdim > 0, "Only enumerate tuples for dimension greater than 0"
    if sdim == 1:
        return [(i, deg - i) for i in range(deg + 1)]
    else:
        return [tuple([i] + list(bt))
                for i in range(deg + 1)
                for bt in berntuples(sdim - 1, deg - i)]


def berntuple_indices(dim, deg):
    return dict([(y, x) for (x, y) in enumerate(berntuples(dim, deg))])


def mi_sum(alpha, beta):
    return tuple([a + b for (a, b) in zip(alpha, beta)])


def barycentric_gradients(verts):
    """Computes the gradients (exterior derivatives) on a cell
       with vertices given in verts.  Writes the result into the
       array grads.  verts is of shape[sd+1, sd], where sd is
       the geometric spatial dimension, and grads is the same shape.
       Note: this assumes that the topological and geometric dimension
       of the cell are the same (e. g. no triangles embedded in 3-space"""
    sd = verts.shape[0]-1
    M = np.zeros((sd+1, sd+1))
    grads = np.zeros((sd+1, sd))
    for p in range(sd+1):
        M[p, 0] = 1.0
        for d in range(sd):
            M[p, d+1] = verts[p, d]
    Minv = np.linalg.inv(M)
    for p in range(sd+1):
        for d in range(sd):
            grads[p, d] = Minv[d+1, p]

    return grads



def gradt_pattern(dim, deg):
    alphas = berntuples(dim, deg)
    alphas_low_indices = berntuple_indices(dim, deg - 1)

    row_ptr = [0]
    scalars = []
    which_lowbern = []
    which_barygrad = []

    for alpha in alphas:
        which_dirs = [i for i in range(len(alpha)) if alpha[i] > 0]
        lower_alphas = [mi_sum(alpha, mei(dim+1, i)) for i in which_dirs]
        lower_alpha_inds = [alphas_low_indices[la] for la in lower_alphas]
        row_ptr.append(row_ptr[-1] + len(which_dirs))
        which_lowbern.extend(lower_alpha_inds)
        which_barygrad.extend(which_dirs)
        scalars.extend([deg] * len(which_dirs))

    return (dim,
            deg-1,
            np.array(row_ptr, dtype=int),
            np.array(scalars, dtype=float),
            np.array(which_lowbern, dtype=int),
            np.array(which_barygrad, dtype=int))


class Bernstein(FiniteElementBase):
    """Scalar - valued Bernstein element. Note: need to work out the
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
            xi = Variable(kernel_data.point_variable_name(points))
            kernel_data.static[static_key] = (xi, lambda: points.points)

        return xi

    def _weights_variable(self, weights, kernel_data):
        """Return the symbolic variables for the static data
        corresponding to the array of weights in a quadrature rule."""

        static_key = (id(weights),)
        if static_key in kernel_data.static:
            wt = kernel_data.static[static_key][0]
        else:
            wt = Variable(kernel_data.weight_variable_name(weights))
            kernel_data.static[static_key] = (wt, lambda: np.array(weights))

        return wt

    @property
    def dofs_shape(self):

        degree = self.degree
        dim = self.cell.get_spatial_dimension()
        return (int(np.prod(xrange(degree + 1, degree + 1 + dim)) /
                    np.prod(xrange(1, dim + 1))),)

    def _grad_pattern(self, kernel_data):
        deg = self.degree
        sd = self.cell.get_spatial_dimension()

        alphas_low = berntuples(sd, deg-1)
        alphas_indices = berntuple_indices(sd, deg)

        sources = np.zeros((len(alphas_low), sd+1), dtype=int)

        for (i, alow) in enumerate(alphas_low):
            for j in range(sd+1):
                atmp = list(alow)
                atmp[j] += 1
                a = tuple(atmp)
                sources[i, j] = alphas_indices[a]

        barygrads = barycentric_gradients(np.array(self.cell.vertices))

        # need to put barygrads & sources into static kernel
        # then the recipe is simple

        sources_static_key = (id(self), "grad_pattern_sources")
        barygrads_static_key = (id(self), "barygrad")

        if sources_static_key in kernel_data.static:
            sources_var = kernel_data.static[sources_static_key][0]
        else:
            sources_var = Variable("bern_grad_sources"+str(id(self)))
            kernel_data.static[sources_static_key] \
                = (sources_var, lambda: sources)

        if barygrads_static_key in kernel_data.static:
            barygrads_var = kernel_data.static[barygrads_static_key][0]
        else:
            barygrads_var = Variable("bern_barygrad"+str(id(self)))
            kernel_data.static[barygrads_static_key] \
                = (barygrads_var, lambda: barygrads)

        return (sources_var, barygrads_var)

    def _gradt_pattern(self, kernel_data):
        deg = self.degree
        sd = self.cell.get_spatial_dimension()
        (sdp, degp, row_ptr, scalars, which_lowbern, which_barygrad) \
            = gradt_pattern(sd, deg)
        barygrads = barycentric_gradients(np.array(self.cell.vertices))

        row_ptr_static_key = (id(self), "row")
        scalars_static_key = (id(self), "scalars")
        which_lowbern_static_key = (id(self), "which_lowbern")
        which_barygrad_static_key = (id(self), "which_barygrad")
        barygrads_static_key = (id(self.cell), "barygrad")

        if row_ptr_static_key in kernel_data.static:
            row_ptr_var = kernel_data.static[row_ptr_static_key][0]
        else:
            row_ptr_var = Variable("bern_grad_row_ptr"+id(self))
            kernel_data.static[row_ptr_static_key] \
                = (row_ptr_var, lambda: row_ptr)
        if scalars_static_key in kernel_data.static:
            scalars_var = kernel_data.static[scalars_static_key][0]
        else:
            scalars_var = Variable("bern_grad_scalars"+id(self))
            kernel_data.static[scalars_static_key] \
                = (scalars_var, lambda: scalars)
        if which_lowbern_static_key in kernel_data.static:
            wlb_var = kernel_data.static[which_lowbern_static_key][0]
        else:
            wlb_var = Variable("bern_which_lowbern"+id(self))
            kernel_data.static[which_lowbern_static_key] \
                = (wlb_var, lambda: which_lowbern)

        if which_barygrad_static_key in kernel_data.static:
            wbg_var = kernel_data.static[which_barygrad_static_key][0]
        else:
            wbg_var = Variable("bern_which_barygrad"+id(self))
            kernel_data.static[which_barygrad_static_key] \
                = (wbg_var, lambda: which_barygrad)

        if barygrads_static_key in kernel_data.static:
            bgs_var = kernel_data[barygrads_static_key][0]
        else:
            bgs_var = Variable("bern_barygrads"+id(self))
            kernel_data.static[barygrads_static_key] \
                = (bgs_var, lambda: barygrads)

        return (row_ptr_var, scalars_var, wlb_var, wbg_var, bgs_var)

    def field_evaluation(self, field_var, q, kernel_data, derivative=None):
        if not isinstance(q.points, StroudPointSet):
            raise ValueError(
                "Only Stroud points may be employed with Bernstein polynomials"
            )

        if derivative is None:
            kernel_data.kernel_args.add(field_var)

            # Get the symbolic names for the points.
            xi = [self._points_variable(f.points, kernel_data)
                  for f in q.factors]

            qs = q.factors

            deg = self.degree

            sd = self.cell.get_spatial_dimension()

            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")
            tmps = [kernel_data.new_variable("tmp") for d in range(sd - 1)]

            # Create basis function indices that run over the possible
            #multiindex space.  These have to be jagged

            alpha = SimpliciallyGradedBasisFunctionIndex(sd, deg)
            alphas = alpha.factors

            # temporary quadrature indices so I don't clobber the ones that
            # have to be free for the entire recipe
            qs_internal = [PointIndex(qf) for qf in q.points.factor_sets]
            
            # For each phase I need to figure out the free variables of
            # that phase
            free_vars_per_phase = []
            for d in range(sd - 1):
                alphas_free_cur = tuple(alphas[:(-1 - d)])
                qs_free_cur = tuple(qs_internal[(-1 - d):])
                free_vars_per_phase.append(((), alphas_free_cur, qs_free_cur))
                # last phase: the free variables are the free quadrature point indices
                free_vars_per_phase.append(((), (), (q,)))
                
            # the first phase reads from the field_var storage
            # the rest of the phases will read from a tmp variable
            # This code computes the offset into the field_var storage
            # for the internal sum variable loop.
            offset = 0
            for d in range(sd - 1):
                deg_begin = deg - mysum(alphas[:d])
                deg_end = deg - alphas[d]
                offset += pd(sd - d, deg_begin) - pd(sd - d, deg_end)

            # each phase of the sum - factored algorithm reads from a particular
            # location.  The first of these is field_var, the rest are the
            # temporaries.
            read_locs = [field_var[alphas[-1] + offset]]
            if sd > 1:
                # intermediate phases will read from the alphas and
                # internal quadrature points
                for d in range(1, sd - 1):
                    tmp_cur = tmps[d - 1]
                    read_alphas = alphas[:(-d)]
                    read_qs = qs_internal[(-d):]
                    read_locs.append(tmp_cur[tuple(read_alphas + read_qs)])

                # last phase reads from the last alpha and the incoming quadrature points
                read_locs.append(tmps[-1][tuple(alphas[:1] + qs[1:])])

            # Figure out the "xi" for each phase being used in the recurrence.
            # In the last phase, it has to refer to one of the free incoming
            # quadrature points, and it refers to the internal ones in previous phases.
            xi_per_phase = [xi[-(d + 1)][qs_internal[-(d + 1)]]
                            for d in range(sd - 1)] + [xi[0][qs[0]]]

            # first phase: no previous phase to bind
            xi_cur = xi_per_phase[0]
            s = 1 - xi_cur
            expr = Let(((r, xi_cur / s),),
                       IndexSum((alphas[-1],),
                                Wave(w,
                                     alphas[-1],
                                     s ** (deg - mysum(alphas[:(sd - 1)])),
                                     w * r * (deg - mysum(alphas) + 1) / (alphas[-1]),
                                     w * read_locs[0]
                                 )
                            )
                   )
            recipe_cur = Recipe(free_vars_per_phase[0], expr)

            for d in range(1, sd):
                # Need to bind the free variables that came before in Let
                # then do what I think is right.
                xi_cur = xi_per_phase[d]
                s = 1 - xi_cur
                alpha_cur = alphas[-(d + 1)]
                asum0 = mysum(alphas[:(sd - d - 1)])
                asum1 = mysum(alphas[:(sd - d)])
                
                expr = Let(((tmps[d - 1], recipe_cur),
                            (r, xi_cur / s)),
                           IndexSum((alpha_cur,),
                                    Wave(w,
                                         alpha_cur,
                                         s ** (deg - asum0),
                                         w * r * (deg - asum1 + 1) / alpha_cur,
                                         w * read_locs[d]
                                     )
                                )
                       )
                recipe_cur = Recipe(free_vars_per_phase[d], expr)

            return recipe_cur
        elif derivative is grad:
            pass
#            a = indices.DimensionIndex(sd)
#            i = indices.BasisFunctionIndex(pd(sd, deg))

#            if deg == 0:
#                return Recipe((a, i, q), 0.0)
#            B_low = Bernstein(self._cell, deg-1)
#            (sources, ref_barygrads) = self._grad_pattern(kernel_data)

            # first stage compute the gradient of the input field
            # loops over lower-degree basis functions and computes
            # its coefficients
#            i = BasisFunctionIndex(pd(sd, deg-1))
#            j = BasisFunctionIndex(sd+1)
#            bg = DimensionIndex(sd)
#
#            gfv_expr = deg * field_var[sources[i, j]] * ref_barygrads[j, bg]
#
#            gfv = Recipe(((bg,), (i,), ()),
#                         IndexSum((j,),
#                                  gfv_expr))
#
#            return B_low.field_evaluation(gfv,
#                                          q,
#                                          kernel_data)
        else:
            raise NotImplementedError

    def moment_evaluation(self, value, weights, q, kernel_data,
                          derivative=None, pullback=None):
        if not isinstance(q.points, StroudPointSet):
            raise ValueError(
                "Only Stroud points may be employed with Bernstein polynomials"
            )

        if derivative is not None:
            raise NotImplementedError

        qs = q.factors

        wt = [self._weights_variable(weights[d], kernel_data)
              for d in range(len(weights))]

        xi = [self._points_variable(f.points, kernel_data)
              for f in q.factors]

        deg = self.degree

        sd = self.cell.get_spatial_dimension()

        r = kernel_data.new_variable("r")
        w = kernel_data.new_variable("w")
        tmps = [kernel_data.new_variable("tmp") for d in range(sd - 1)]

        if sd == 2:
            alpha_internal = SimpliciallyGradedBasisFunctionIndex(sd, deg)
            alphas_int = alpha_internal.factors
            xi_cur = xi[0][qs[1]]
            s = 1 - xi_cur
            expr0 = Let(((r, xi_cur / s), ),
                        IndexSum((qs[0], ),
                                 Wave(w,
                                      alphas_int[0],
                                      wt[0][qs[0]] * (s ** deg),
                                      w * r * (deg - alphas_int[0]) / alphas_int[0],
                                      w * value[qs[0], qs[1]])
                                 )
                        )
            expr0prime = ForAll((qs[1],),
                                ForAll((alphas_int[0],),
                                       expr0))
            recipe0 = Recipe(((), (alphas_int[0], ), (qs[1], )),
                             expr0prime)
            xi_cur = xi[1]
            s = 1 - xi_cur
            alpha = SimpliciallyGradedBasisFunctionIndex(2, deg)
            alphas = alpha.factors
            r = xi_cur / s
            expr1 = Let(((tmps[0], recipe0), ),
                        IndexSum((qs[1], ),
                                 ForAll((alphas[0],),
                                        ForAll((alphas[1],),
                                               Wave(w,
                                                    alphas[1],
                                                    wt[1][qs[1]] * (s ** (deg - alphas[0])),
                                                    w * r * (deg - alphas[0] - alphas[1] + 1) / (alphas[1]),
                                                    w * tmps[0][alphas[0], qs[1]])
                                               )
                                        )
                                 )
                        )

            return Recipe(((), (alphas[0], alphas[1]), ()), expr1)

        else:
            raise NotImplementedError

    def moment_evaluation_general(self, value, weights, q, kernel_data,
                                  derivative=None, pullback=None):
        if not isinstance(q.points, StroudPointSet):
            raise ValueError("Only Stroud points may be employed with Bernstein polynomials")

        if derivative is not None:
            raise NotImplementedError

        qs = q.factors

        wt = [self._weights_variable(weights[d], kernel_data)
              for d in range(len(weights))]

        xi = [self._points_variable(f.points, kernel_data)
              for f in q.factors]

        deg = self.degree

        sd = self.cell.get_spatial_dimension()

        r = kernel_data.new_variable("r")
        w = kernel_data.new_variable("w")
        tmps = [kernel_data.new_variable("tmp") for d in range(sd - 1)]

        # the output recipe is parameterized over these
        alpha = SimpliciallyGradedBasisFunctionIndex(sd, deg)
        alphas = alpha.factors

        read_locs = [value[q]]
        for d in range(1, sd - 1):
            tmp_cur = tmps[d - 1]
            read_alphas = alphas[:d]
            read_qs = qs[-d:]
            read_locs.append(tmp_cur[tuple(read_alphas + read_qs)])
        d = sd - 1
        tmp_cur = tmps[d - 1]
        read_alphas = alphas[:d]
        read_qs = qs[-d:]
        read_locs.append(tmp_cur[tuple(read_alphas + read_qs)])

        free_vars_per_phase = []
        for d in range(1, sd):
            alphas_free_cur = tuple(alphas[:d])
            qs_free_cur = tuple(qs[-d:])
            free_vars_per_phase.append(((), alphas_free_cur, qs_free_cur))
        free_vars_per_phase.append(((), (), tuple(alphas)))

        xi_cur = xi[0][qs[0]]
        s = 1 - xi_cur
        expr = Let(((r, xi_cur / s),),
                   IndexSum((qs[0],),
                            Wave(w,
                                 alphas[0],
                                 wt[0][qs[0]] * s ** deg,
                                 w * r * (deg - alphas[0] - 1) / alphas[0],
                                 w * read_locs[0]
                                 )
                            )
                   )

        recipe_cur = Recipe(free_vars_per_phase[0], expr)

        for d in range(1, sd):
            xi_cur = xi[d]
            s = 1 - xi_cur
            acur = alphas[d]
            asum0 = mysum(alphas[:d])
            asum1 = asum0 - acur
            expr = Let(((tmps[d - 1], recipe_cur),
                        (r, xi_cur / s)),
                       IndexSum((qs[d],),
                                Wave(w,
                                     acur,
                                     wt[d][qs[d]] * s ** (deg - asum0),
                                     w * r * (deg - asum1 - 1) / acur,
                                     w * read_locs[d]
                                     )
                                )
                       )
            recipe_cur = Recipe(free_vars_per_phase[d], expr)

        return recipe_cur


# We're doing a 1d case to get all the field evaluation cases right
# first, then will put the higher dimension back in
class Bernstein1D(FiniteElementBase):
    def __init__(self, cell, degree):
        super(Bernstein1D, self).__init__()

        self._cell = cell
        self._degree = degree

    def _points_variable(self, points, kernel_data):
        """Return the symbolic variables for the static data
        corresponding to the :class:`PointSet` ``points``."""

        static_key = (id(points),)
        if static_key in kernel_data.static:
            xi = kernel_data.static[static_key][0]
        else:
            xi = Variable(kernel_data.point_variable_name(points))
            kernel_data.static[static_key] = (xi, lambda: points.points)

        return xi

    def _weights_variable(self, weights, kernel_data):
        """Return the symbolic variables for the static data
        corresponding to the array of weights in a quadrature rule."""

        static_key = (id(weights),)
        if static_key in kernel_data.static:
            wt = kernel_data.static[static_key][0]
        else:
            wt = Variable(kernel_data.weight_variable_name(weights))
            kernel_data.static[static_key] = (wt, lambda: np.array(weights))

        return wt

    def _grad_pattern(self, kernel_data):
        deg = self.degree
        sd = self.cell.get_spatial_dimension()
        alphas_low = berntuples(sd, deg-1)
        alphas_indices = berntuple_indices(sd, deg)

        sources = np.zeros((len(alphas_low), sd+1), dtype=int)

        for (i, alow) in enumerate(alphas_low):
            for j in range(sd+1):
                atmp = list(alow)
                atmp[j] += 1
                a = tuple(atmp)
                sources[i, sd-j] = alphas_indices[a]

        barygrads = barycentric_gradients(np.array(self.cell.vertices))

        # need to put barygrads & sources into static kernel
        # then the recipe is simple

        sources_static_key = (id(self), "grad_pattern_sources")
        barygrads_static_key = (id(self), "barygrad")

        if sources_static_key in kernel_data.static:
            sources_var = kernel_data.static[sources_static_key][0]
        else:
            sources_var = Variable("bern_grad_sources"+str(id(self)))
            kernel_data.static[sources_static_key] \
                = (sources_var, lambda: sources)

        if barygrads_static_key in kernel_data.static:
            barygrads_var = kernel_data.static[barygrads_static_key][0]
        else:
            barygrads_var = Variable("bern_barygrad"+str(id(self)))
            kernel_data.static[barygrads_static_key] \
                = (barygrads_var, lambda: barygrads)

        return (sources_var, barygrads_var)

    @property
    def dofs_shape(self):

        degree = self.degree
        dim = self.cell.get_spatial_dimension()
        return (int(np.prod(xrange(degree + 1, degree + 1 + dim)) /
                    np.prod(xrange(1, dim + 1))),)

    def gradient(self, field_var, kernel_data):
        """Returns the recipe for computing the gradient coefficients,
        given the field_var coefficients."""
        deg = self.degree

        if not isinstance(field_var, Variable):
            raise ValueError("gradient only works on variables")

        ii = BasisFunctionIndex(deg)
        kk = DimensionIndex(2)
        jj = DimensionIndex(1)

        (sources, ref_barygrads) = self._grad_pattern(kernel_data)

        grad_bod = deg * IndexSum((kk,),
                                  field_var[sources[ii, kk]]
                                  * ref_barygrads[kk, jj])

        return Recipe(((jj,), (ii,), ()), grad_bod)

    def field_evaluation(self, field_var, q, kernel_data,
                         derivative=None):
        # Note: point types don't matter in 1d since there's no
        # tensor structure used

        deg = self.degree

        if derivative is grad:
            if field_var.__class__ != Variable:
                raise ValueError("Can't differentiate that")
            if deg == 0:
                i = BasisFunctionIndex(deg)
                return Recipe(((DimensionIndex(1),), (i,), ()), 0.0)
            else:
                grad_field_var = self.gradient(field_var, kernel_data)
                B_low = Bernstein1D(self._cell, deg-1)
                return B_low.field_evaluation(grad_field_var, q, kernel_data)
        elif derivative is None:
            if isinstance(field_var, Recipe):
                fv = field_var
            elif field_var.__class__ == Variable:
                i = BasisFunctionIndex(deg+1)
                fv = Recipe(((), (i,), ()), field_var[i])
            else:
                raise ValueError("Illegal input field")

            d_fv, b_fv, p_fv = fv.indices

            assert len(b_fv) == 1

            (i,) = b_fv

            # We should only have one basis function index in the field
            r = kernel_data.new_variable("r")
            w = kernel_data.new_variable("w")

            # q is a point iterates over the points.
            # I need to figure out how to get the points themselves!
            flat_points = PointSet(q.points.points()[:, 0])
            q_flat = PointIndex(flat_points)
            xi = flat_points.kernel_variable("xi", kernel_data)
            s = 1 - xi[q_flat]
            
            expr = Let(((r, xi[q_flat]/s),),
                       IndexSum((i,),
                                Wave(w,
                                     i,
                                     s**deg,
                                     w*r*(deg-i+1) / i,
                                     w * fv[i])))
            return Recipe((d_fv, (), p_fv+(q_flat,)), expr)
        else:
            raise NotImplementedError

    def moment_evaluation(self, value, weights, q,
                          kernel_data, derivative=None,
                          pullback=True):
        if derivative is grad:
            pass
        elif not derivative:
            if isinstance(value, Recipe):
                pass
            elif value.__class__ == Variable:
                pass
            else:
                raise ValueError("Illegal input values")
        else:
            raise NotImplementedError
