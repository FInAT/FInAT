from itertools import chain, repeat

import gem
import numpy
import symengine
import sympy
from FIAT.polynomial_set import mis
from FIAT.reference_element import UFCQuadrilateral

from finat.finiteelementbase import FiniteElementBase
from finat.physically_mapped import Citations, DirectlyDefinedElement
from finat.sympy2gem import sympy2gem


class DirectSerendipity(DirectlyDefinedElement, FiniteElementBase):
    def __init__(self, cell, degree):
        if Citations is not None:
            Citations().register("Arbogast2017")

        # These elements only known currently on quads
        assert isinstance(cell, UFCQuadrilateral)

        self._cell = cell
        self._degree = degree

    @property
    def cell(self):
        return self._cell

    @property
    def degree(self):
        return self._degree

    @property
    def formdegree(self):
        return 0

    def entity_dofs(self):
        if self.degree == 1:
            return {0: {i: [i] for i in range(4)},
                    1: {i: [] for i in range(4)},
                    2: {0: []}}
        elif self.degree == 2:
            return {0: {i: [i] for i in range(4)},
                    1: {i: [i+4] for i in range(4)},
                    2: {0: []}}
        else:
            return {0: {i: [i] for i in range(4)},
                    1: {i: list(range(4 + i * (self.degree-1),
                                      4 + (i + 1) * (self.degree-1)))
                        for i in range(4)},
                    2: {0: list(range(4 + 4 * (self.degree - 1),
                                      self.space_dimension()))}}

    def space_dimension(self):
        return 4 if self.degree == 1 else (self.degree+1)*(self.degree+2)//2 + 2

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    @property
    def value_shape(self):
        return ()

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''
        ct = self.cell.topology

        # Build everything in sympy
        vs, xx, phis = ds_sympy(ct, self.degree)
        if self.degree == 1:
            vs, xx, phis = ds1_sympy(ct)
        else:
            vs, xx, phis = dsr_sympy(ct, self.degree)

        # and convert -- all this can be used for each derivative!
        phys_verts = coordinate_mapping.physical_vertices()

        phys_points = gem.partial_indexed(
            coordinate_mapping.physical_points(ps, entity=entity),
            ps.indices)

        repl = {vs[i, j]: phys_verts[i, j] for i in range(4) for j in range(2)}

        repl.update({s: phys_points[i] for i, s in enumerate(xx)})

        mapper = gem.node.Memoizer(sympy2gem)
        mapper.bindings = repl

        result = {}

        for i in range(order+1):
            alphas = mis(2, i)
            for alpha in alphas:
                dphis = [diff(phi, xx, alpha) for phi in phis]
                result[alpha] = gem.ListTensor(list(map(mapper, dphis)))

        return result

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError("Not done yet, sorry!")

    def mapping(self):
        return "physical"


def xysub(x, y):
    return {x[0]: y[0], x[1]: y[1]}


def ds1_sympy(ct, vs=None, sp=sympy):
    """Constructs lowest-order case of Arbogast's directly defined C^0 serendipity
    elements, which are a special case.
    :param ct: The cell topology of the reference quadrilateral.
    :param vs: (Optional) coordinates of cell on which to construct the basis.
               If it is None, this function constructs symbols for the vertices.
    :returns: a 3-tuple containing symbols for the physical cell coordinates and the
              physical cell independent variables (e.g. "x" and "y") and a list
              of the four basis functions.
    """
    if vs is None:
        vs = numpy.asarray(list(zip(symengine.symbols('x:4'),
                                    symengine.symbols('y:4'))))
    else:
        vs = numpy.asarray(vs)

    xx = numpy.asarray(symengine.symbols("x,y"))

    ts = numpy.zeros((4, 2), dtype=object)
    for e in range(4):
        v0id, v1id = ct[1][e][:]
        for j in range(2):
            ts[e, :] = vs[v1id, :] - vs[v0id, :]

    ns = numpy.zeros((4, 2), dtype=object)
    for e in (0, 3):
        ns[e, 0] = -ts[e, 1]
        ns[e, 1] = ts[e, 0]

    for e in (1, 2):
        ns[e, 0] = ts[e, 1]
        ns[e, 1] = -ts[e, 0]

    xstars = numpy.zeros((4, 2), dtype=object)
    for e in range(4):
        v0id, v1id = ct[1][e][:]
        xstars[e, :] = (vs[v0id, :] + vs[v1id])/2

    lams = [(xx-xstars[i, :]) @ ns[i, :] for i in range(4)]

    RV = (lams[0] - lams[1]) / (lams[0] + lams[1])
    RH = (lams[2] - lams[3]) / (lams[2] + lams[3])
    Rs = [RV, RH]

    xis = []
    for e in range(4):
        dct = xysub(xx, xstars[e, :])
        i = 2*((3-e)//2)
        j = i + 1
        xi = lams[i] * lams[j] * (1 + (-1)**(e+1) * Rs[e//2]) / lams[i].subs(dct) / lams[j].subs(dct) / 2
        xis.append(xi)

    d = xysub(xx, vs[0, :])
    r = lams[1] * lams[3] / lams[1].subs(d) / lams[3].subs(d)
    d = xysub(xx, vs[2, :])
    r -= lams[0] * lams[3] / lams[0].subs(d) / lams[3].subs(d)
    d = xysub(xx, vs[3, :])
    r += lams[0] * lams[2] / lams[0].subs(d) / lams[2].subs(d)
    d = xysub(xx, vs[1, :])
    r -= lams[1] * lams[2] / lams[1].subs(d) / lams[2].subs(d)
    R = r - sum([r.subs(xysub(xx, xstars[i, :])) * xis[i] for i in range(4)])

    n03 = numpy.array([[0, -1], [1, 0]]) @ (vs[3, :] - vs[0, :])
    lam03 = (xx - vs[0, :]) @ n03
    n12 = numpy.array([[0, -1], [1, 0]]) @ (vs[2, :] - vs[1, :])
    lam12 = (xx - vs[2, :]) @ n12

    phi0tilde = lam12 - lam12.subs({xx[0]: vs[3, 0], xx[1]: vs[3, 1]}) * (1 + R) / 2
    phi1tilde = lam03 - lam03.subs({xx[0]: vs[2, 0], xx[1]: vs[2, 1]}) * (1 - R) / 2
    phi2tilde = lam03 - lam03.subs({xx[0]: vs[1, 0], xx[1]: vs[1, 1]}) * (1 - R) / 2
    phi3tilde = lam12 - lam12.subs({xx[0]: vs[0, 0], xx[1]: vs[0, 1]}) * (1 + R) / 2

    phis = []
    for i, phitilde in enumerate([phi0tilde, phi1tilde, phi2tilde, phi3tilde]):
        phi = phitilde / phitilde.subs({xx[0]: vs[i, 0], xx[1]: vs[i, 1]})
        phis.append(phi)

    return vs, xx, numpy.asarray(phis)


def newton_dd(nds, fs):
    """Constructs Newton's divided differences for the input arrays,
    which may include symbolic values."""
    n = len(nds)
    mat = numpy.zeros((n, n), dtype=object)
    mat[:, 0] = fs[:]
    for j in range(1, n):
        for i in range(n-j):
            mat[i, j] = (mat[i+1, j-1] - mat[i, j-1]) / (nds[i+j] - nds[i])
    return mat[0, :]


def newton_poly(nds, fs, xsym):
    """Constructs  Lagrange interpolating polynomial passing through
    x values nds and y values fs.  Returns a a symbolic object in terms
    of independent variable xsym."""
    coeffs = newton_dd(nds, fs)
    result = coeffs[-1]
    n = len(coeffs)
    for i in range(n-2, -1, -1):
        result = result * (xsym - nds[i]) + coeffs[i]
    return result


def diff(expr, xx, alpha):
    """Differentiate expr with respect to xx.

    :arg expr: symengine/symengine Expression to differentiate.
    :arg xx: iterable of coordinates to differentiate with respect to.
    :arg alpha: derivative multiindex, one entry for each entry of xx
        indicating how many derivatives in that direction.
    :returns: New symengine/symengine expression."""
    if isinstance(expr, sympy.Expr):
        return expr.diff(*(zip(xx, alpha)))
    else:
        return symengine.diff(expr, *(chain(*(repeat(x, a) for x, a in zip(xx, alpha)))))


def dsr_sympy(ct, r, vs=None, sp=sympy):
    """Constructs higher-order (>= 2) case of Arbogast's directly defined C^0 serendipity
    elements, which include all polynomials of degree r plus a couple of rational
    functions.
    :param ct: The cell topology of the reference quadrilateral.
    :param vs: (Optional) coordinates of cell on which to construct the basis.
               If it is None, this function constructs symbols for the vertices.
    :returns: a 3-tuple containing symbols for the physical cell coordinates and the
              physical cell independent variables (e.g. "x" and "y") and a list
              of the four basis functions.
    """
    if vs is None:  # do vertices symbolically
        vs = numpy.asarray(list(zip(sp.symbols('x:4'),
                                    sp.symbols('y:4'))))
    else:
        vs = numpy.asarray(vs)
    xx = numpy.asarray(sp.symbols("x,y"))

    ts = numpy.zeros((4, 2), dtype=object)
    for e in range(4):
        v0id, v1id = ct[1][e][:]
        ts[e, :] = vs[v1id, :] - vs[v0id, :]

    ns = numpy.zeros((4, 2), dtype=object)
    for e in (0, 3):
        ns[e, 0] = -ts[e, 1]
        ns[e, 1] = ts[e, 0]

    for e in (1, 2):
        ns[e, 0] = ts[e, 1]
        ns[e, 1] = -ts[e, 0]

    # midpoints of each edge
    xstars = numpy.zeros((4, 2), dtype=object)
    for e in range(4):
        v0id, v1id = ct[1][e][:]
        xstars[e, :] = (vs[v0id, :] + vs[v1id])/2

    lams = [(xx-xstars[i, :]) @ ns[i, :] for i in range(4)]

    # # internal functions
    bubble = numpy.prod(lams)

    if r < 4:
        internal_bfs = []
        internal_nodes = []
    elif r == 4:  # Just one point
        xbar = sum(vs[i, 0] for i in range(4)) / 4
        ybar = sum(vs[i, 1] for i in range(4)) / 4
        internal_bfs = [bubble / bubble.subs(xysub(xx, (xbar, ybar)))]
        internal_nodes = [(xbar, ybar)]
    else:  # build a lattice inside the quad
        dx0 = (vs[1, :] - vs[0, :]) / (r-2)
        dx1 = (vs[2, :] - vs[0, :]) / (r-2)

        internal_nodes = [vs[0, :] + dx0 * i + dx1 * j
                          for i in range(1, r-2)
                          for j in range(1, r-1-i)]

        mons = [xx[0] ** i * xx[1] ** j
                for i in range(r-3) for j in range(r-3-i)]

        V = sp.Matrix([[mon.subs(xysub(xx, nd)) for mon in mons]
                       for nd in internal_nodes])
        Vinv = V.inv()
        nmon = len(mons)

        internal_bfs = []
        for j in range(nmon):
            preibf = bubble * sum(Vinv[i, j] * mons[i] for i in range(nmon))
            internal_bfs.append(preibf
                                / preibf.subs(xysub(xx, internal_nodes[j])))

    RV = (lams[0] - lams[1]) / (lams[0] + lams[1])
    RH = (lams[2] - lams[3]) / (lams[2] + lams[3])

    # R for each edge (1 on edge, zero on opposite
    Rs = [(1 - RV) / 2, (1 + RV) / 2, (1 - RH) / 2, (1 + RH) / 2]

    nodes1d = [sp.Rational(i, r) for i in range(1, r)]

    s = sp.Symbol('s')

    # for each edge:
    # I need its adjacent two edges
    # and its opposite edge
    # and its "tunnel R" RH or RV
    # This is very 2d specific.
    opposite_edges = {e: [eother for eother in ct[1]
                          if set(ct[1][e]).intersection(ct[1][eother]) == set()][0]
                      for e in ct[1]}
    adjacent_edges = {e: tuple(sorted([eother for eother in ct[1]
                                       if eother != e
                                       and set(ct[1][e]).intersection(ct[1][eother])
                                       != set()]))
                      for e in ct[1]}

    ae = adjacent_edges
    tunnel_R_edges = {e: ((lams[ae[e][0]] - lams[ae[e][1]])
                          / (lams[ae[e][0]] + lams[ae[e][1]]))
                      for e in range(4)}
    edge_nodes = []
    for ed in range(4):
        ((v0x, v0y), (v1x, v1y)) = vs[ct[1][ed], :]
        delx = v1x - v0x
        dely = v1y - v0y
        edge_nodes.append([(v0x+nd*delx, v0y+nd*dely) for nd in nodes1d])

    # subtracts off the value of function at internal nodes times those
    # internal basis functions
    def nodalize(f):
        return f - sum(f.subs(xysub(xx, nd)) * bf
                       for bf, nd in zip(internal_bfs, internal_nodes))

    edge_bfs = []
    if r == 2:
        for ed in range(4):
            lamadj0 = lams[adjacent_edges[ed][0]]
            lamadj1 = lams[adjacent_edges[ed][1]]
            ephi = lamadj0 * lamadj1 * Rs[ed]
            phi = nodalize(ephi) / ephi.subs(xysub(xx, xstars[ed]))
            edge_bfs.append([phi])
    else:
        for ed in range(4):
            ((v0x, v0y), (v1x, v1y)) = vs[ct[1][ed], :]
            Rcur = tunnel_R_edges[ed]
            lam_op = lams[opposite_edges[ed]]

            edge_bfs_cur = []

            for i in range(len(nodes1d)):
                # strike out i:th node
                idcs = [j for j in range(len(nodes1d)) if i != j]
                nodes1d_cur = [nodes1d[j] for j in idcs]
                edge_nodes_cur = [edge_nodes[ed][j]
                                  for j in idcs]

                # construct the 1d interpolation with remaining nodes
                pvals = []
                for nd in edge_nodes_cur:
                    sub = xysub(xx, nd)
                    pval_cur = (-1 * Rcur.subs(sub)**(r-2)
                                / lam_op.subs(sub))
                    pvals.append(pval_cur)

                ptilde = newton_poly(nodes1d_cur, pvals, s)
                xt = xx @ ts[ed]
                vt0 = numpy.asarray((v0x, v0y)) @ ts[ed]
                vt1 = numpy.asarray((v1x, v1y)) @ ts[ed]
                p = ptilde.subs({s: (xt-vt0) / (vt1-vt0)})

                prebf = (lams[adjacent_edges[ed][0]]
                         * lams[adjacent_edges[ed][1]]
                         * (lams[opposite_edges[ed]] * p
                            + Rcur**(r-2) * Rs[ed]))

                prebf = nodalize(prebf)
                bfcur = prebf / prebf.subs(xysub(xx, edge_nodes[ed][i]))
                edge_bfs_cur.append(bfcur)

            edge_bfs.append(edge_bfs_cur)

    # vertex basis functions
    vertex_to_adj_edges = {i: tuple([e for e in ct[1] if i in ct[1][e]])
                           for i in ct[0]}
    vertex_to_off_edges = {i: tuple([e for e in ct[1] if i not in ct[1][e]])
                           for i in ct[0]}

    vertex_bfs = []
    for v in range(4):
        ed0, ed1 = vertex_to_off_edges[v]
        lam0 = lams[ed0]
        lam1 = lams[ed1]

        prebf = lam0 * lam1

        # subtract off edge values
        for adj_ed in vertex_to_adj_edges[v]:
            edge_nodes_cur = edge_nodes[adj_ed]
            edge_bfs_cur = edge_bfs[adj_ed]
            for k, (nd, edbf) in enumerate(zip(edge_nodes_cur, edge_bfs_cur)):
                sb = xysub(xx, nd)
                prebf -= lam0.subs(sb) * lam1.subs(sb) * edbf

        bf = nodalize(prebf) / prebf.subs(xysub(xx, vs[v, :]))
        vertex_bfs.append(bf)

    bfs = vertex_bfs
    for edbfs in edge_bfs:
        bfs.extend(edbfs)
    bfs.extend(internal_bfs)

    nds = [tuple(vs[i, :]) for i in range(4)]
    for ends in edge_nodes:
        nds.extend(ends)
    nds.extend(internal_nodes)

    return vs, xx, numpy.asarray(bfs)


def ds_sympy(ct, r, vs=None, sp=sympy):
    """Symbolically Constructs Arbogast's directly defined C^0 serendipity elements,
    which include all polynomials of degree r plus a couple of rational functions.
    :param ct: The cell topology of the reference quadrilateral.
    :param vs: (Optional) coordinates of cell on which to construct the basis.
               If it is None, this function constructs symbols for the vertices.
    :returns: a 3-tuple containing symbols for the physical cell coordinates and the
              physical cell independent variables (e.g. "x" and "y") and a list
              of the four basis functions.
    """
    if r == 1:
        return ds1_sympy(ct, vs, sp=sp)
    else:
        return dsr_sympy(ct, r, vs, sp=sp)
