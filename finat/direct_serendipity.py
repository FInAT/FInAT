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
        assert degree == 1 or degree == 2
        self._cell = cell
        self._degree = degree
        self.space_dim = 4 if degree == 1 else (self.degree+1)*(self.degree+2)/2 + 2

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
        else:
            return {0: {i: [i] for i in range(4)},
                    1: {i: [i+4] for i in range(4)},
                    2: {0: []}}

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
        elif self.degree == 2:
            vs, xx, phis = ds2_sympy(ct)

        # and convert -- all this can be used for each derivative!
        phys_verts = coordinate_mapping.physical_vertices()

        phys_points = gem.partial_indexed(coordinate_mapping.physical_points(ps, entity=entity),
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

    def xysub(x, y):
        return {x[0]: y[0], x[1]: y[1]}

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


def ds2_sympy(ct):
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

    def xysub(x, y):
        return {x[0]: y[0], x[1]: y[1]}

    xstars = numpy.zeros((4, 2), dtype=object)
    for e in range(4):
        v0id, v1id = ct[1][e][:]
        xstars[e, :] = (vs[v0id, :] + vs[v1id])/2

    lams = [(xx-xstars[i, :]) @ ns[i, :] for i in range(4)]

    RV = (lams[0] - lams[1]) / (lams[0] + lams[1])
    RH = (lams[2] - lams[3]) / (lams[2] + lams[3])

    xx2xstars = [xysub(xx, xstars[i]) for i in range(4)]
    v_phitildes = [(lams[1] * lams[3]
                    - lams[3].subs(xx2xstars[2])
                    / lams[0].subs(xx2xstars[2])
                    * lams[0] * lams[1] * (1-RH) / 2
                    - lams[1].subs(xx2xstars[0])
                    / lams[2].subs(xx2xstars[0])
                    * lams[2] * lams[3] * (1-RV) / 2),
                   (lams[1] * lams[2]
                    - lams[2].subs(xx2xstars[3])
                    / lams[0].subs(xx2xstars[3])
                    * lams[0] * lams[1] * (1+RH) / 2
                    - lams[1].subs(xx2xstars[0])
                    / lams[3].subs(xx2xstars[0])
                    * lams[2] * lams[3] * (1-RV) / 2),
                   (lams[0] * lams[3]
                    - lams[3].subs(xx2xstars[2])
                    / lams[1].subs(xx2xstars[2])
                    * lams[0] * lams[1] * (1-RH) / 2
                    - lams[0].subs(xx2xstars[1])
                    / lams[2].subs(xx2xstars[1])
                    * lams[2] * lams[3] * (1+RV) / 2),
                   (lams[0] * lams[2]
                    - lams[2].subs(xx2xstars[3])
                    / lams[1].subs(xx2xstars[3])
                    * lams[0] * lams[1] * (1+RH) / 2
                    - lams[0].subs(xx2xstars[1])
                    / lams[3].subs(xx2xstars[1])
                    * lams[2] * lams[3] * (1+RV) / 2)]
    phis_v = [phitilde_v / phitilde_v.subs(xysub(xx, vs[i, :]))
              for i, phitilde_v in enumerate(v_phitildes)]

    e_phitildes = [lams[2] * lams[3] * (1-RV) / 2,
                   lams[2] * lams[3] * (1+RV) / 2,
                   lams[0] * lams[1] * (1-RH) / 2,
                   lams[0] * lams[1] * (1+RH) / 2]

    phis_e = [ephi / ephi.subs(xx2xstars[i]) for i, ephi in enumerate(e_phitildes)]

    return vs, xx, numpy.asarray(phis_v + phis_e)
