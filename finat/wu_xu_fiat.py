# Copyright (C) 2022 Robert C. Kirby (Baylor University)
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.

from FIAT import (expansions, polynomial_set, quadrature, dual_set,
                  finite_element, functional, Bubble, Lagrange,
                  reference_element, ufc_simplex)
import numpy
from collections import OrderedDict

polydim = expansions.polynomial_dimension


class IntegralMomentOfNormalDerivative(functional.Functional):
    """Functional giving normal derivative integrated on a facet."""

    def __init__(self, ref_el, facet_no, Q, avg=True):
        # Average -- leaves off edge weight, effectively
        # dividing the moment by it.
        n = ref_el.compute_normal(facet_no)
        self.n = n
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        # map points onto facet

        fmap = ref_el.get_entity_transform(sd-1, facet_no)
        qpts, qwts = Q.get_points(), Q.get_weights()
        if not avg:
            qwts *= ref_el.volume_of_subcomplex(1, facet_no)
        dpts = [fmap(pt) for pt in qpts]
        self.dpts = dpts

        dpt_dict = OrderedDict()

        alphas = [[1 if j == i else 0 for j in range(sd)] for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*n[i], tuple(alphas[i]), tuple()) for i in range(sd)]

        functional.Functional.__init__(
            self, ref_el, tuple(),
            {}, dpt_dict, "IntegralMomentOfNormalDerivative")


class IntegralMomentOfTangentialDerivative(functional.Functional):
    """Functional giving tangential derivative integrated on a facet."""

    def __init__(self, ref_el, facet_no, Q, avg=True):
        t = ref_el.compute_normalized_edge_tangent(facet_no)
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        # map points onto facet

        fmap = ref_el.get_entity_transform(sd-1, facet_no)
        qpts, qwts = Q.get_points(), Q.get_weights()
        if not avg:
            qwts *= ref_el.volume_of_subcomplex(1, facet_no)

        dpts = [fmap(pt) for pt in qpts]
        self.dpts = dpts

        dpt_dict = OrderedDict()

        # quadrature weights don't include Jacobian, but
        # normal does for "scaled" normal, otherwise, we've divided
        # the integral by the length to get the average.

        alphas = [[1 if j == i else 0 for j in range(sd)] for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*t[i], tuple(alphas[i]), tuple()) for i in range(sd)]

        functional.Functional.__init__(
            self, ref_el, tuple(),
            {}, dpt_dict, "IntegralMomentOfTangentialDerivative")


class IntegralMomentOfSecondNormalDerivative(functional.Functional):
    """Functional giving second normal derivative integrated on a facet."""

    def __init__(self, ref_el, facet_no, Q, avg=True):
        n = ref_el.compute_normal(facet_no)
        sd = ref_el.get_spatial_dimension()

        self.n = n
        self.Q = Q

        nu = [n[i]**2 for i in range(sd)] + [2.0*n[i]*n[j] for i in range(sd) for j in range(i+1, sd)]

        alphas = []
        nu = []
        for i in range(sd):
            for j in range(i, sd):
                alpha = [0] * sd
                alpha[i] += 1
                alpha[j] += 1
                alphas.append(alpha)
                if i != j:
                    nu.append(2.0*n[i]*n[j])
                else:
                    nu.append(n[i]*n[j])

        # map points onto facet

        fmap = ref_el.get_entity_transform(sd-1, facet_no)
        qpts, qwts = Q.get_points(), Q.get_weights()
        qpts, qwts = Q.get_points(), Q.get_weights()
        if not avg:
            qwts *= ref_el.volume_of_subcomplex(1, facet_no)
        dpts = [fmap(pt) for pt in qpts]
        self.dpts = dpts

        dpt_dict = OrderedDict()

        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*nui, tuple(alpha), tuple()) for nui, alpha in zip(nu, alphas)]

        functional.Functional.__init__(
            self, ref_el, tuple(),
            {}, dpt_dict, "IntegralMomentOfSecondNormalDerivative")


class IntegralMomentOfSecondTangentialDerivative(functional.Functional):
    """Functional giving second tangential derivative integrated on a facet."""

    def __init__(self, ref_el, facet_no, Q, avg=True):
        t = ref_el.compute_normalized_edge_tangent(facet_no)
        self.t = t
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        alphas = []
        tau = []
        for i in range(sd):
            for j in range(i, sd):
                alpha = [0] * sd
                alpha[i] += 1
                alpha[j] += 1
                alphas.append(alpha)
                if i != j:
                    tau.append(2.0*t[i]*t[j])
                else:
                    tau.append(t[i]*t[j])

        # map points onto facet

        fmap = ref_el.get_entity_transform(sd-1, facet_no)
        qpts, qwts = Q.get_points(), Q.get_weights()
        if not avg:
            qwts *= ref_el.volume_of_subcomplex(1, facet_no)

        dpts = [fmap(pt) for pt in qpts]
        self.dpts = dpts

        dpt_dict = OrderedDict()

        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*taui, tuple(alpha), tuple()) for taui, alpha in zip(tau, alphas)]

        functional.Functional.__init__(
            self, ref_el, tuple(),
            {}, dpt_dict, "IntegralMomentOfSecondTangentialDerivative")


class IntegralMomentOfSecondMixedDerivative(functional.Functional):
    """Functional giving second mixed derivative integrated on a facet."""

    def __init__(self, ref_el, facet_no, Q, avg=True):
        n = ref_el.compute_normal(facet_no)
        t = ref_el.compute_normalized_edge_tangent(facet_no)

        sd = ref_el.get_spatial_dimension()

        self.n = n
        self.t = t
        self.Q = Q

        alphas = []
        nutau = []
        for i in range(sd):
            for j in range(sd):
                alpha = [0] * sd
                alpha[i] += 1
                alpha[j] += 1
                alphas.append(alpha)
                nutau.append(n[i]*t[j])

        # map points onto facet

        fmap = ref_el.get_entity_transform(sd-1, facet_no)
        qpts, qwts = Q.get_points(), Q.get_weights()
        if not avg:
            qwts *= ref_el.volume_of_subcomplex(1, facet_no)
        dpts = [fmap(pt) for pt in qpts]
        self.dpts = dpts

        dpt_dict = OrderedDict()

        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*nui, tuple(alpha), tuple()) for nui, alpha in zip(nutau, alphas)]

        functional.Functional.__init__(
            self, ref_el, tuple(), {}, dpt_dict,
            "IntegralMomentOfSecondNormalDerivative")


def WuXuRobustH3NCSpace(ref_el):
    """Constructs a basis for the the Wu Xu H^3 nonconforming space
    P^{(3,2)}(T) = P_3(T) + b_T P_1(T) + b_T^2 P_1(T),
    where b_T is the standard cubic bubble."""

    sd = ref_el.get_spatial_dimension()
    assert sd == 2

    em_deg = 7

    # Unfortunately,  b_T^2 P_1 has degree 7 (cubic squared times a linear)
    # so we need a high embedded degree!
    p7 = polynomial_set.ONPolynomialSet(ref_el, 7)

    dimp1 = polydim(ref_el, 1)
    dimp3 = polydim(ref_el, 3)
    dimp7 = polydim(ref_el, 7)

    # Here's the first bit we'll work with.  It's already expressed in terms
    # of the ON basis for P7, so we're golden.
    p3fromp7 = p7.take(list(range(dimp3)))

    # Rather than creating the barycentric coordinates ourself, let's
    # reuse the existing bubble functionality
    bT = Bubble(ref_el, 3)
    p1 = Lagrange(ref_el, 1)

    # next, we'll have to project b_T P1 and b_T^2 P1 onto P^7
    Q = quadrature.make_quadrature(ref_el, 8)
    Qpts = numpy.array(Q.get_points())
    Qwts = numpy.array(Q.get_weights())

    zero_index = tuple([0 for i in range(sd)])

    # it's just one bubble function: let's get a 1d array!
    bT_at_qpts = bT.tabulate(0, Qpts)[zero_index][0, :]
    p1_at_qpts = p1.tabulate(0, Qpts)[zero_index]

    # Note: difference in signature because bT, p1 are FE and p7 is a
    # polynomial set
    p7_at_qpts = p7.tabulate(Qpts)[zero_index]

    bubble_coeffs = numpy.zeros((6, dimp7), "d")

    # first three: bT P1, last three will be bT^2 P1
    foo = bT_at_qpts * p1_at_qpts * Qwts
    bubble_coeffs[:dimp1, :] = numpy.dot(foo, p7_at_qpts.T)

    foo = bT_at_qpts * foo
    bubble_coeffs[dimp1:2*dimp1, :] = numpy.dot(foo, p7_at_qpts.T)

    bubbles = polynomial_set.PolynomialSet(ref_el, 3, em_deg,
                                           p7.get_expansion_set(),
                                           bubble_coeffs,
                                           p7.get_dmats())

    return polynomial_set.polynomial_set_union_normalized(p3fromp7, bubbles)


def WuXuH3NCSpace(ref_el):
    """Constructs a basis for the the Wu Xu H^3 nonconforming space
    P(T) = P_3(T) + b_T P_1(T),
    where b_T is the standard cubic bubble."""

    assert ref_el.get_spatial_dimension() == 2

    em_deg = 4
    p4 = polynomial_set.ONPolynomialSet(ref_el, em_deg)

    # Here's the first bit we'll work with.  It's already expressed in terms
    # of the ON basis for P4, so we're golden.
    p3fromp4 = p4.take(list(range(10)))

    # Rather than creating the barycentric coordinates ourself, let's
    # reuse the existing bubble functionality
    bT = Bubble(ref_el, 4)

    return polynomial_set.polynomial_set_union_normalized(p3fromp4, bT.get_nodal_basis())


class WuXuRobustH3NCDualSet(dual_set.DualSet):
    """Dual basis for WuXu H3 nonconforming element consisting of
    vertex values and gradients and first and second normals at edge midpoints."""

    def __init__(self, ref_el, avg=True):
        entity_ids = {}
        nodes = []
        cur = 0

        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        assert sd == 2

        pe = functional.PointEvaluation
        pd = functional.PointDerivative
        eind = IntegralMomentOfNormalDerivative
        eindd = IntegralMomentOfSecondNormalDerivative

        # jet at each vertex

        entity_ids[0] = {}
        for v in sorted(top[0]):
            # point value
            nodes.append(pe(ref_el, verts[v]))

            # gradient
            for i in range(sd):
                alpha = [0]*sd
                alpha[i] = 1
                nodes.append(pd(ref_el, verts[v], alpha))

            entity_ids[0][v] = list(range(cur, cur+1+sd))
            cur += sd + 1

        entity_ids[1] = {}

        # quadrature rule for edge integrals
        Q = quadrature.make_quadrature(ufc_simplex(1), 4)

        for e in sorted(top[1]):
            n = eind(ref_el, e, Q, avg)
            nodes.extend([n])
            entity_ids[1][e] = [cur]
            cur += 1

        for e in sorted(top[1]):
            nn = eindd(ref_el, e, Q, avg)
            nodes.extend([nn])
            entity_ids[1][e].append(cur)
            cur += 1

        entity_ids[2] = {0: []}

        super(WuXuRobustH3NCDualSet, self).__init__(nodes, ref_el, entity_ids)


class WuXuH3NCDualSet(dual_set.DualSet):
    """Dual basis for WuXu H3 nonconforming element consisting of
    vertex values and gradients and second normals at edge midpoints."""

    def __init__(self, ref_el, avg=True):
        entity_ids = {}
        nodes = []
        cur = 0

        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        assert sd == 2

        pe = functional.PointEvaluation
        pd = functional.PointDerivative
        eindd = IntegralMomentOfSecondNormalDerivative

        # jet at each vertex
        entity_ids[0] = {}
        for v in sorted(top[0]):
            # point value
            nodes.append(pe(ref_el, verts[v]))

            # gradient
            for i in range(sd):
                alpha = [0]*sd
                alpha[i] = 1
                nodes.append(pd(ref_el, verts[v], alpha))

            entity_ids[0][v] = list(range(cur, cur+1+sd))
            cur += sd + 1

        entity_ids[1] = {}

        # quadrature rule for edge integrals
        Q = quadrature.make_quadrature(ufc_simplex(1), 4)
        for e in sorted(top[1]):
            nn = eindd(ref_el, e, Q, avg)
            nodes.extend([nn])
            entity_ids[1][e] = [cur]
            cur += 1

        entity_ids[2] = {0: []}

        super(WuXuH3NCDualSet, self).__init__(nodes, ref_el, entity_ids)


class WuXuRobustH3NC(finite_element.CiarletElement):
    """The Wu-Xu robust H3 nonconforming finite element"""
    def __init__(self, ref_el, avg=True):
        poly_set = WuXuRobustH3NCSpace(ref_el)
        dual = WuXuRobustH3NCDualSet(ref_el, avg)
        super(WuXuRobustH3NC, self).__init__(poly_set, dual, 7)


class WuXuH3NC(finite_element.CiarletElement):
    """The Wu-Xu H3 nonconforming finite element"""
    def __init__(self, ref_el, avg=True):
        poly_set = WuXuH3NCSpace(ref_el)
        dual = WuXuH3NCDualSet(ref_el, avg)
        super(WuXuH3NC, self).__init__(poly_set, dual, 4)


def getD(phys_el):
    D = numpy.zeros((18, 12))
    ns = numpy.array(
        [phys_el.compute_normal(e) for e in range(3)])
    ts = numpy.array(
        [phys_el.compute_normalized_edge_tangent(e) for e in range(3)])

    for j, i in enumerate(list(range(10)) + [12, 15]):
        D[i, j] = 1.0

    D[10, 4:6] = -ns[0, :]
    D[10, 7:9] = ns[0, :]
    D[11, 4:6] = -ts[0, :]
    D[11, 7:9] = ts[0, :]
    D[13, 1:3] = -ns[1, :]
    D[13, 7:9] = ns[1, :]
    D[14, 1:3] = -ts[1, :]
    D[14, 7:9] = ts[1, :]
    D[16, 1:3] = -ns[2, :]
    D[16, 4:6] = ns[2, :]
    D[17, 1:3] = -ts[2, :]
    D[17, 4:6] = ts[2, :]

    return D


def getD_robust(phys_el):
    D = numpy.zeros((24, 15))

    ns = numpy.array(
        [phys_el.compute_normal(e) for e in range(3)])

    ts = numpy.array(
        [phys_el.compute_normalized_edge_tangent(e) for e in range(3)])

    for j, i in enumerate(list(range(10)) + [11, 13, 15, 18, 21]):
        D[i, j] = 1.0

    D[10, 3] = -1.0
    D[10, 6] = 1.0
    D[12, 0] = -1.0
    D[12, 6] = 1.0
    D[14, 0] = -1.0
    D[14, 3] = 1.0

    D[16, 4] = -ns[0, 0]
    D[16, 5] = -ns[0, 1]
    D[16, 7] = ns[0, 0]
    D[16, 8] = ns[0, 1]
    D[17, 4] = -ts[0, 0]
    D[17, 5] = -ts[0, 1]
    D[17, 7] = ts[0, 0]
    D[17, 8] = ts[0, 1]
    D[19, 1] = -ns[1, 0]
    D[19, 2] = -ns[1, 1]
    D[19, 7] = ns[1, 0]
    D[19, 8] = ns[1, 1]
    D[20, 1] = -ts[1, 0]
    D[20, 2] = -ts[1, 1]
    D[20, 7] = ts[1, 0]
    D[20, 8] = ts[1, 1]
    D[22, 1] = -ns[2, 0]
    D[22, 2] = -ns[2, 1]
    D[22, 4] = ns[2, 0]
    D[22, 5] = ns[2, 1]
    D[23, 1] = -ts[2, 0]
    D[23, 2] = -ts[2, 1]
    D[23, 4] = ts[2, 0]
    D[23, 5] = ts[2, 1]

    return D


def transform_robust(Tphys, That):
    lens = numpy.array([Tphys.volume_of_subcomplex(1, e)
                        for e in range(3)])

    # do I need to reverse this in FInAT?
    J, b = reference_element.make_affine_mapping(Tphys.vertices, That.vertices)
    Jinv = numpy.linalg.inv(J)
    [[dxdxhat, dxdyhat], [dydxhat, dydyhat]] = Jinv

    Thetainv = numpy.array(
        [[dxdxhat**2, 2 * dxdxhat * dydxhat, dydxhat**2],
         [dxdyhat * dxdxhat, dxdyhat * dydxhat + dxdxhat * dydyhat, dydxhat * dydyhat],
         [dxdyhat**2, 2 * dxdyhat * dydyhat, dydyhat**2]])

    # extract actual nodes from extended set.
    E = numpy.zeros((15, 24))
    for i in range(9):
        E[i, i] = 1
    E[9, 9] = 1
    E[10, 11] = 1
    E[11, 13] = 1
    E[12, 15] = 1
    E[13, 18] = 1
    E[14, 21] = 1

    Vc = numpy.zeros((24, 24))

    # let's build geometric things for each edge
    ns = numpy.zeros((3, 2))
    nhats = numpy.zeros((3, 2))
    ts = numpy.zeros((3, 2))
    thats = numpy.zeros((3, 2))
    Gs = numpy.zeros((3, 2, 2))
    Ghats = numpy.zeros((3, 2, 2))

    Gammas = numpy.zeros((3, 3, 3))
    Gammahats = numpy.zeros((3, 3, 3))
    B1s = numpy.zeros((3, 2, 2))
    B2s = numpy.zeros((3, 3, 3))

    for e in range(3):
        nhats[e, :] = That.compute_normal(e)
        ns[e, :] = Tphys.compute_normal(e)
        thats[e, :] = That.compute_normalized_edge_tangent(e)
        ts[e, :] = Tphys.compute_normalized_edge_tangent(e)
        Gs[e, :, :] = numpy.asarray([ns[e, :], ts[e, :]])
        Ghats[e, :, :] = numpy.asarray([nhats[e, :], thats[e, :]])

        nx = ns[e, 0]
        ny = ns[e, 1]
        tx = ts[e, 0]
        ty = ts[e, 1]
        nhatx = nhats[e, 0]
        nhaty = nhats[e, 1]
        thatx = thats[e, 0]
        thaty = thats[e, 1]

        Gammas[e, :, :] = numpy.asarray(
            [[nx**2, 2*nx*tx, tx**2],
             [nx*ny, nx*ty+ny*tx, tx*ty],
             [ny**2, 2*ny*ty, ty**2]])

        Gammahats[e, :, :] = numpy.asarray(
            [[nhatx**2, 2*nhatx*thatx, thatx**2],
             [nhatx*nhaty, nhatx*thaty+nhaty*thatx, thatx*thaty],
             [nhaty**2, 2*nhaty*thaty, thaty**2]])

        dt = numpy.dot
        inv = numpy.linalg.inv
        B1s[e, :, :] = dt(Ghats[e], dt(Jinv.T, Gs[e].T)) / lens[e]
        B2s[e, :, :] = dt(inv(Gammahats[e]), dt(Thetainv, Gammas[e])) / lens[e]

    for e in range(3):
        Vc[3*e, 3*e] = 1.0
        Vc[3*e+1:3*e+3, 3*e+1:3*e+3] = Jinv.T

    for e in range(3):
        Vc[9+2*e:9+2*e+2, 9+2*e:9+2*e+2] = B1s[e]
        Vc[15+3*e:15+3*(e+1), 15+3*e:15+3*(e+1)] = B2s[e]

    D = getD_robust(Tphys)

    V = dt(E, dt(Vc, D))

    return (E, Vc, D, V)


def transform(Tphys, That):
    lens = numpy.array([Tphys.volume_of_subcomplex(1, e)
                        for e in range(3)])

    J, b = reference_element.make_affine_mapping(Tphys.vertices, That.vertices)
    Jinv = numpy.linalg.inv(J)
    [[dxdxhat, dxdyhat], [dydxhat, dydyhat]] = Jinv

    Thetainv = numpy.array(
        [[dxdxhat**2, 2 * dxdxhat * dydxhat, dydxhat**2],
         [dxdyhat * dxdxhat, dxdyhat * dydxhat + dxdxhat * dydyhat, dydxhat * dydyhat],
         [dxdyhat**2, 2 * dxdyhat * dydyhat, dydyhat**2]])

    # extract actual nodes from extended set.
    E = numpy.zeros((12, 18))
    for i in range(9):
        E[i, i] = 1
    E[9, 9] = 1
    E[10, 12] = 1
    E[11, 15] = 1

    Vc = numpy.zeros((18, 18))

    # let's build geometric things for each edge
    ns = numpy.zeros((3, 2))
    nhats = numpy.zeros((3, 2))
    ts = numpy.zeros((3, 2))
    thats = numpy.zeros((3, 2))
    Gs = numpy.zeros((3, 2, 2))
    Ghats = numpy.zeros((3, 2, 2))

    Gammas = numpy.zeros((3, 3, 3))
    Gammahats = numpy.zeros((3, 3, 3))
    B1s = numpy.zeros((3, 2, 2))
    B2s = numpy.zeros((3, 3, 3))

    for e in range(3):
        nhats[e, :] = That.compute_normal(e)
        ns[e, :] = Tphys.compute_normal(e)
        thats[e, :] = That.compute_normalized_edge_tangent(e)
        ts[e, :] = Tphys.compute_normalized_edge_tangent(e)
        Gs[e, :, :] = numpy.asarray([ns[e, :], ts[e, :]])
        Ghats[e, :, :] = numpy.asarray([nhats[e, :], thats[e, :]])

        nx = ns[e, 0]
        ny = ns[e, 1]
        tx = ts[e, 0]
        ty = ts[e, 1]
        nhatx = nhats[e, 0]
        nhaty = nhats[e, 1]
        thatx = thats[e, 0]
        thaty = thats[e, 1]

        Gammas[e, :, :] = numpy.asarray(
            [[nx**2, 2*nx*tx, tx**2],
             [nx*ny, nx*ty+ny*tx, tx*ty],
             [ny**2, 2*ny*ty, ty**2]])

        Gammahats[e, :, :] = numpy.asarray(
            [[nhatx**2, 2*nhatx*thatx, thatx**2],
             [nhatx*nhaty, nhatx*thaty+nhaty*thatx, thatx*thaty],
             [nhaty**2, 2*nhaty*thaty, thaty**2]])

        dt = numpy.dot
        inv = numpy.linalg.inv
        B1s[e, :, :] = dt(Ghats[e], dt(Jinv.T, Gs[e].T)) / lens[e]
        B2s[e, :, :] = dt(inv(Gammahats[e]), dt(Thetainv, Gammas[e])) / lens[e]

    for e in range(3):
        Vc[3*e, 3*e] = 1.0
        Vc[3*e+1:3*e+3, 3*e+1:3*e+3] = Jinv.T

    for e in range(3):
        # Vc[9+2*e:9+2*e+2, 9+2*e:9+2*e+2] = B1s[e]
        Vc[9+3*e:9+3*(e+1), 9+3*e:9+3*(e+1)] = B2s[e]

    D = getD(Tphys)

    V = dt(E, dt(Vc, D))

    return (E, Vc, D, V)


def compact_transform_robust(Tphys, That):
    lens = numpy.array([Tphys.volume_of_subcomplex(1, e)
                        for e in range(3)])

    # do I need to reverse this in FInAT?
    J, b = reference_element.make_affine_mapping(Tphys.vertices, That.vertices)
    Jinv = numpy.linalg.inv(J)
    [[dxdxhat, dxdyhat], [dydxhat, dydyhat]] = Jinv

    Thetainv = numpy.array(
        [[dxdxhat**2, 2 * dxdxhat * dydxhat, dydxhat**2],
         [dxdyhat * dxdxhat, dxdyhat * dydxhat + dxdxhat * dydyhat, dydxhat * dydyhat],
         [dxdyhat**2, 2 * dxdyhat * dydyhat, dydyhat**2]])

    ns = numpy.zeros((3, 2))
    nhats = numpy.zeros((3, 2))
    ts = numpy.zeros((3, 2))
    thats = numpy.zeros((3, 2))
    Gs = numpy.zeros((3, 2, 2))
    Ghats = numpy.zeros((3, 2, 2))

    Gammas = numpy.zeros((3, 3, 3))
    Gammahats = numpy.zeros((3, 3, 3))
    B1s = numpy.zeros((3, 2, 2))
    B2s = numpy.zeros((3, 3, 3))
    betas = numpy.zeros((3, 2))

    for e in range(3):
        nhats[e, :] = That.compute_normal(e)
        ns[e, :] = Tphys.compute_normal(e)
        thats[e, :] = That.compute_normalized_edge_tangent(e)
        ts[e, :] = Tphys.compute_normalized_edge_tangent(e)
        Gs[e, :, :] = numpy.asarray([ns[e, :], ts[e, :]])
        Ghats[e, :, :] = numpy.asarray([nhats[e, :], thats[e, :]])

        nx = ns[e, 0]
        ny = ns[e, 1]
        tx = ts[e, 0]
        ty = ts[e, 1]
        nhatx = nhats[e, 0]
        nhaty = nhats[e, 1]
        thatx = thats[e, 0]
        thaty = thats[e, 1]

        Gammas[e, :, :] = numpy.asarray(
            [[nx**2, 2*nx*tx, tx**2],
             [nx*ny, nx*ty+ny*tx, tx*ty],
             [ny**2, 2*ny*ty, ty**2]])

        Gammahats[e, :, :] = numpy.asarray(
            [[nhatx**2, 2*nhatx*thatx, thatx**2],
             [nhatx*nhaty, nhatx*thaty+nhaty*thatx, thatx*thaty],
             [nhaty**2, 2*nhaty*thaty, thaty**2]])

        dt = numpy.dot
        inv = numpy.linalg.inv
        B1s[e, :, :] = dt(Ghats[e], dt(Jinv.T, Gs[e].T)) / lens[e]
        B2s[e, :, :] = dt(inv(Gammahats[e]), dt(Thetainv, Gammas[e])) / lens[e]

        betas[e, 0] = nx * B2s[e, 0, 1] + tx * B2s[e, 0, 2]
        betas[e, 1] = ny * B2s[e, 0, 1] + ty * B2s[e, 0, 2]

    V = numpy.zeros((15, 15))
    for e in range(3):
        V[3*e, 3*e] = 1.0
        V[3*e+1:3*e+3, 3*e+1:3*e+3] = Jinv.T

    V[10, 0] = -B1s[1, 0, 1]
    V[11, 0] = -B1s[2, 0, 1]
    V[9, 3] = -B1s[0, 0, 1]
    V[11, 3] = B1s[2, 0, 1]
    V[9, 6] = B1s[0, 0, 1]
    V[10, 6] = B1s[1, 0, 1]

    for e in range(9, 12):
        V[e, e] = B1s[e-9, 0, 0]

    for e in range(12, 15):
        V[e, e] = B2s[e-12, 0, 0]

    V[13, 1:3] = -betas[1, :]
    V[14, 1:3] = -betas[2, :]
    V[12, 4:6] = -betas[0, :]
    V[14, 4:6] = betas[2, :]
    V[12, 7:9] = betas[0, :]
    V[13, 7:9] = betas[1, :]

    return V


def compact_transform(Tphys, That):
    lens = numpy.array([Tphys.volume_of_subcomplex(1, e)
                        for e in range(3)])

    # do I need to reverse this in FInAT?
    J, b = reference_element.make_affine_mapping(Tphys.vertices, That.vertices)
    Jinv = numpy.linalg.inv(J)
    [[dxdxhat, dxdyhat], [dydxhat, dydyhat]] = Jinv

    Thetainv = numpy.array(
        [[dxdxhat**2, 2 * dxdxhat * dydxhat, dydxhat**2],
         [dxdyhat * dxdxhat, dxdyhat * dydxhat + dxdxhat * dydyhat, dydxhat * dydyhat],
         [dxdyhat**2, 2 * dxdyhat * dydyhat, dydyhat**2]])

    ns = numpy.zeros((3, 2))
    nhats = numpy.zeros((3, 2))
    ts = numpy.zeros((3, 2))
    thats = numpy.zeros((3, 2))
    Gs = numpy.zeros((3, 2, 2))
    Ghats = numpy.zeros((3, 2, 2))

    Gammas = numpy.zeros((3, 3, 3))
    Gammahats = numpy.zeros((3, 3, 3))
    B1s = numpy.zeros((3, 2, 2))
    B2s = numpy.zeros((3, 3, 3))
    betas = numpy.zeros((3, 2))

    for e in range(3):
        nhats[e, :] = That.compute_normal(e)
        ns[e, :] = Tphys.compute_normal(e)
        thats[e, :] = That.compute_normalized_edge_tangent(e)
        ts[e, :] = Tphys.compute_normalized_edge_tangent(e)
        Gs[e, :, :] = numpy.asarray([ns[e, :], ts[e, :]])
        Ghats[e, :, :] = numpy.asarray([nhats[e, :], thats[e, :]])

        nx = ns[e, 0]
        ny = ns[e, 1]
        tx = ts[e, 0]
        ty = ts[e, 1]
        nhatx = nhats[e, 0]
        nhaty = nhats[e, 1]
        thatx = thats[e, 0]
        thaty = thats[e, 1]

        Gammas[e, :, :] = numpy.asarray(
            [[nx**2, 2*nx*tx, tx**2],
             [nx*ny, nx*ty+ny*tx, tx*ty],
             [ny**2, 2*ny*ty, ty**2]])

        Gammahats[e, :, :] = numpy.asarray(
            [[nhatx**2, 2*nhatx*thatx, thatx**2],
             [nhatx*nhaty, nhatx*thaty+nhaty*thatx, thatx*thaty],
             [nhaty**2, 2*nhaty*thaty, thaty**2]])

        dt = numpy.dot
        inv = numpy.linalg.inv
        B1s[e, :, :] = dt(Ghats[e], dt(Jinv.T, Gs[e].T)) / lens[e]
        B2s[e, :, :] = dt(inv(Gammahats[e]), dt(Thetainv, Gammas[e])) / lens[e]

        betas[e, 0] = nx * B2s[e, 0, 1] + tx * B2s[e, 0, 2]
        betas[e, 1] = ny * B2s[e, 0, 1] + ty * B2s[e, 0, 2]

    V = numpy.zeros((12, 12))
    for e in range(3):
        V[3*e, 3*e] = 1.0
        V[3*e+1:3*e+3, 3*e+1:3*e+3] = Jinv.T

    for e in range(9, 12):
        V[e, e] = B2s[e-9, 0, 0]

    V[9, 4:6] = -betas[0, :]
    V[9, 7:9] = betas[0, :]
    V[10, 1:3] = -betas[1, :]
    V[10, 7:9] = betas[1, :]
    V[11, 1:3] = -betas[2, :]
    V[11, 4:6] = betas[2, :]

    return V


def test_robust():
    Tref = reference_element.ufc_simplex(2)
    Tphys = reference_element.ufc_simplex(2)
    Tphys.vertices = ((0.0, 0.0), (0.5, 0.1), (0, 0.4))

    WX = WuXuRobustH3NC(Tphys, False)
    WXhat = WuXuRobustH3NC(Tref, True)

    (E, Vc, D, V) = transform_robust(Tphys, Tref)

    Vfoo = compact_transform_robust(Tphys, Tref)

    print(numpy.allclose(V, Vfoo))

    ref_pts = reference_element.make_lattice(Tref.vertices, 4, 1)
    phys_pts = reference_element.make_lattice(Tphys.vertices, 4, 1)

    phys_vals = WX.tabulate(0, phys_pts)[0, 0]
    ref_vals = WXhat.tabulate(0, ref_pts)[0, 0]

    diffs = phys_vals - numpy.dot(V.T, ref_vals)

    if not numpy.allclose(diffs, numpy.zeros(diffs.shape)):
        for i in range(15):
            for j in range(len(ref_pts)):
                if numpy.abs(diffs[i, j]) < 1.e-10:
                    diffs[i, j] = 0.0
    else:
        print("It works!")


def test():
    Tref = reference_element.ufc_simplex(2)
    Tphys = reference_element.ufc_simplex(2)
    Tphys.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    print("Edge lengths:")
    for e in range(3):
        print(Tphys.volume_of_subcomplex(1, e))
    
    WX = WuXuH3NC(Tphys, False)
    WXhat = WuXuH3NC(Tref, True)

    (E, Vc, D, V) = transform(Tphys, Tref)

    Vfoo = compact_transform(Tphys, Tref)

    assert numpy.allclose(V, Vfoo)

    ref_pts = Tref.make_points(2, 0, 6)
    phys_pts = Tphys.make_points(2, 0, 6)

    phys_vals = WX.tabulate(0, phys_pts)[0, 0]
    ref_vals = WXhat.tabulate(0, ref_pts)[0, 0]

    assert numpy.allclose(phys_vals, V.T @ ref_vals)

if __name__ == "__main__":
    # test_robust()
    test()
# 2.1829567105190155
# 1.7464535493393458
# 1.1853269591129698
