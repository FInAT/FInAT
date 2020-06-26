import numpy

from finat.finiteelementbase import FiniteElementBase
from finat.physically_mapped import DirectlyDefinedElement
from FIAT.reference_element import UFCQuadrilateral
from FIAT.polynomial_set import mis

import gem
import sympy
from finat.sympy2gem import sympy2gem


class DirectSerendipity(DirectlyDefinedElement, FiniteElementBase):
    def __init__(self, cell, degree):
        assert isinstance(cell, UFCQuadrilateral)
        # assert degree == 1 or degree == 2
        self._cell = cell
        self._degree = degree
        self.space_dim = 4 if degree == 1 else (self.degree+1)*(self.degree+2)//2 + 2

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
                                      self.space_dim))}}

    def space_dimension(self):
        return self.space_dim

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
                dphis = [phi.diff(*tuple(zip(xx, alpha))) for phi in phis]
                result[alpha] = gem.ListTensor(list(map(mapper, dphis)))

        return result

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError("Not done yet, sorry!")

    def mapping(self):
        return "physical"


def xysub(x, y):
    return {x[0]: y[0], x[1]: y[1]}


def ds1_sympy(ct):
    vs = numpy.asarray(list(zip(sympy.symbols('x:4'), sympy.symbols('y:4'))))
    xx = numpy.asarray(sympy.symbols("x,y"))

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
    n = len(nds)
    mat = numpy.zeros((n, n), dtype=object)
    mat[:, 0] = fs[:]
    for j in range(1, n):
        for i in range(n-j):
            mat[i, j] = (mat[i+1, j-1] - mat[i, j-1]) / (nds[i+j] - nds[i])
    return mat[0, :]


# Horner evaluation of polynomial in symbolic form
def newton_poly(nds, fs, xsym):
    coeffs = newton_dd(nds, fs)
    result = coeffs[-1]
    n = len(coeffs)
    for i in range(n-2, -1, -1):
        result = result * (xsym - nds[i]) + coeffs[i]
    return result


def dsr_sympy(ct, r, vs=None):
    if vs is None:
        vs = numpy.asarray(list(zip(sympy.symbols('x:4'),
                                    sympy.symbols('y:4'))))
    xx = numpy.asarray(sympy.symbols("x,y"))

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

        V = sympy.Matrix([[mon.subs(xysub(xx, nd)) for mon in mons]
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

    nodes1d = [sympy.Rational(i, r) for i in range(1, r)]

    s = sympy.Symbol('s')

    # for each edge:
    # I need its adjacent two edges
    # and its opposite edge
    # and its "tunnel R" RH or RV
    opposite_edges = {0: 1, 1: 0, 2: 3, 3: 2}
    adjacent_edges = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}
    tunnel_R_edges = {0: RH, 1: RH, 2: RV, 3: RV}

    edge_nodes = []
    for ed in range(4):
        ((v0x, v0y), (v1x, v1y)) = vs[ct[1][ed], :]
        delx = v1x - v0x
        dely = v1y - v0y
        edge_nodes.append([(v0x+nd*delx, v0y+nd*dely) for nd in nodes1d])

    # subtracts off the value of function at internal nodes times those
    # internal basis functions
    def nodalize(f):
        foo = f
        for (bf, nd) in zip(internal_bfs, internal_nodes):
            foo = foo - f.subs(xx, nd) * bf
        return foo

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

                bfcur = (nodalize(prebf)
                         / prebf.subs(xysub(xx, edge_nodes[ed][i])))
                edge_bfs_cur.append(bfcur)

            edge_bfs.append(edge_bfs_cur)

    # vertex basis functions
    vertex_to_adj_edges = {0: (0, 2), 1: (0, 3), 2: (1, 2), 3: (1, 3)}
    vertex_to_off_edges = {0: (1, 3), 1: (1, 2), 2: (0, 3), 3: (0, 2)}
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
