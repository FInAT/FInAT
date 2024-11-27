import numpy
from math import comb

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement


def _vertex_transform(V, vorder, fiat_cell, coordinate_mapping):
    """Basis transformation for evaluation, gradient, and hessian at vertices."""
    sd = fiat_cell.get_spatial_dimension()
    top = fiat_cell.get_topology()
    bary, = fiat_cell.make_points(sd, 0, sd+1)
    J = coordinate_mapping.jacobian_at(bary)

    gdofs = sd
    G = [[J[j, i] for j in range(sd)] for i in range(sd)]

    if vorder < 2:
        hdofs = 0
        H = [[]]
    else:
        hdofs = (sd*(sd+1))//2
        indices = [(i, j) for i in range(sd) for j in range(i, sd)]
        H = numpy.zeros((hdofs, hdofs), dtype=object)
        for p, (i, j) in enumerate(indices):
            for q, (m, n) in enumerate(indices):
                H[p, q] = J[m, i] * J[n, j] + J[m, j] * J[n, i]
        H[:, [i == j for i, j in indices]] *= 0.5

    s = 0
    for v in sorted(top[0]):
        s += 1
        V[s:s+gdofs, s:s+gdofs] = G
        s += gdofs
        V[s:s+hdofs, s:s+hdofs] = H
        s += hdofs
    return V


def _normal_tangential_transform(fiat_cell, J, detJ, f):
    R = numpy.array([[0, 1], [-1, 0]])
    that = fiat_cell.compute_edge_tangent(f)
    nhat = R @ that
    Jn = J @ Literal(nhat)
    Jt = J @ Literal(that)
    alpha = Jn @ Jt
    beta = Jt @ Jt
    Bnn = detJ / beta
    Bnt = alpha / beta

    Lhat = numpy.linalg.norm(that)
    Bnn = Bnn * Lhat
    Bnt = Bnt / Lhat
    return Bnn, Bnt, Jt


def _edge_transform(V, vorder, eorder, fiat_cell, coordinate_mapping, avg=False):
    """Basis transformation for integral edge moments.

    :arg V: the transpose of the basis transformation.
    :arg vorder: the jet order at vertices, matching the Jacobi weights in the
                 normal derivative moments on edges.
    :arg eorder: the order of the normal derivative moments.
    :arg fiat_cell: the reference triangle.
    :arg coordinate_mapping: the coordinate mapping.
    :kwarg avg: are we scaling integrals by dividing by the edge length?
    """
    sd = fiat_cell.get_spatial_dimension()
    bary, = fiat_cell.make_points(sd, 0, sd+1)
    J = coordinate_mapping.jacobian_at(bary)
    detJ = coordinate_mapping.detJ_at(bary)
    pel = coordinate_mapping.physical_edge_lengths()

    # number of DOFs per vertex/edge
    voffset = comb(sd + vorder, vorder)
    eoffset = 2 * eorder + 1
    top = fiat_cell.get_topology()
    for e in sorted(top[1]):
        Bnn, Bnt, Jt = _normal_tangential_transform(fiat_cell, J, detJ, e)
        if avg:
            Bnn = Bnn * pel[e]

        v0id, v1id = (v * voffset for v in top[1][e])
        s0 = len(top[0]) * voffset + e * eoffset
        for k in range(eorder+1):
            s = s0 + k
            # Jacobi polynomial at the endpoints
            P1 = comb(k + vorder, k)
            P0 = -(-1)**k * P1
            V[s, s] = Bnn
            V[s, v1id] = P1 * Bnt
            V[s, v0id] = P0 * Bnt
            if k > 0:
                V[s, s + eorder] = -1 * Bnt


class Argyris(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5, variant=None, avg=False):
        if Citations is not None:
            Citations().register("Argyris1968")
        if variant is None:
            variant = "integral"
        if variant == "point" and degree != 5:
            raise NotImplementedError("Degree must be 5 for 'point' variant of Argyris")
        fiat_element = FIAT.Argyris(cell, degree, variant=variant)
        self.variant = variant
        self.avg = avg
        super().__init__(fiat_element)

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()

        V = identity(self.space_dimension())

        vorder = 2
        voffset = comb(sd + vorder, vorder)
        eorder = self.degree - 5

        _vertex_transform(V, vorder, self.cell, coordinate_mapping)
        if self.variant == "integral":
            _edge_transform(V, vorder, eorder, self.cell, coordinate_mapping, avg=self.avg)
        else:
            bary, = self.cell.make_points(sd, 0, sd+1)
            J = coordinate_mapping.jacobian_at(bary)
            detJ = coordinate_mapping.detJ_at(bary)
            pel = coordinate_mapping.physical_edge_lengths()
            for e in sorted(top[1]):
                s = len(top[0]) * voffset + e * (eorder+1)
                v0id, v1id = (v * voffset for v in top[1][e])
                Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e)

                # edge midpoint normal derivative
                V[s, s] = Bnn * pel[e]

                # vertex points
                V[s, v1id] = 15/8 * Bnt
                V[s, v0id] = -1 * V[s, v1id]

                # vertex derivatives
                for i in range(sd):
                    V[s, v1id+1+i] = -7/16 * Bnt * Jt[i]
                    V[s, v0id+1+i] = V[s, v1id+1+i]

                # second derivatives
                tau = [Jt[0]*Jt[0], 2*Jt[0]*Jt[1], Jt[1]*Jt[1]]
                for i in range(len(tau)):
                    V[s, v1id+3+i] = 1/32 * Bnt * tau[i]
                    V[s, v0id+3+i] = -1 * V[s, v1id+3+i]

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            s = voffset*v + 1
            V[:, s:s+sd] *= 1 / h[v]
            V[:, s+sd:voffset*(v+1)] *= 1 / (h[v]*h[v])

        if self.variant == "point":
            eoffset = 2 * eorder + 1
            for e in sorted(top[1]):
                v0, v1 = top[1][e]
                s = len(top[0]) * voffset + e * eoffset
                V[:, s:s+eorder+1] *= 2 / (h[v0] + h[v1])

        return ListTensor(V.T)
