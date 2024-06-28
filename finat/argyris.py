import numpy
from math import comb

import FIAT

from gem import Literal, ListTensor, partial_indexed

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


def _edge_transform(V, voffset, fiat_cell, moment_deg, coordinate_mapping, avg=False):
    """Basis transformation for integral edge moments."""

    J = coordinate_mapping.jacobian_at([1/3, 1/3])
    rns = coordinate_mapping.reference_normals()
    pts = coordinate_mapping.physical_tangents()
    pns = coordinate_mapping.physical_normals()
    pel = coordinate_mapping.physical_edge_lengths()

    top = fiat_cell.get_topology()
    eoffset = 2 * moment_deg - 1
    for e in sorted(top[1]):
        nhat = partial_indexed(rns, (e, ))
        n = partial_indexed(pns, (e, ))
        t = partial_indexed(pts, (e, ))
        Bn = J @ nhat / pel[e]
        Bnt = Bn @ t
        Bnn = Bn @ n
        if avg:
            Bnn = Bnn * pel[e]

        v0id, v1id = (v * voffset for v in top[1][e])
        s0 = len(top[0]) * voffset + e * eoffset
        toffset = s0 + moment_deg
        V[s0, s0] = Bnn
        V[s0, v1id] = Bnt
        V[s0, v0id] = -1 * Bnt
        for k in range(1, moment_deg):
            s = s0 + k
            P1 = Literal(comb(k + 3, k))
            P0 = (-1)**(k-1) * P1
            V[s, s] = Bnn
            V[s, v1id] = P1 * Bnt
            V[s, v0id] = P0 * Bnt
            V[s, toffset + k-1] = -1 * Bnt


class Argyris(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree, variant=None, avg=False):
        if Citations is not None:
            Citations().register("Argyris1968")
        if variant is None:
            variant = "integral"
        if variant == "integral":
            fiat_element = FIAT.Argyris(cell, degree, variant=variant)
        elif variant == "point":
            if degree != 5:
                raise ValueError("Degree must be 5 for 'point' variant of Argyris")
            fiat_element = FIAT.QuinticArgyris(cell)
        else:
            raise ValueError("Invalid variant for Argyris")
        self.variant = variant
        self.avg = avg
        super().__init__(fiat_element)

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at edge midpoints
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        ndof = self.space_dimension()
        V = numpy.eye(ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        voffset = (sd+1)*sd//2 + sd + 1
        for v in sorted(top[0]):
            s = voffset * v
            for i in range(sd):
                for j in range(sd):
                    V[s+1+i, s+1+j] = J[j, i]
            V[s+3, s+3] = J[0, 0]*J[0, 0]
            V[s+3, s+4] = 2*J[0, 0]*J[1, 0]
            V[s+3, s+5] = J[1, 0]*J[1, 0]
            V[s+4, s+3] = J[0, 0]*J[0, 1]
            V[s+4, s+4] = J[0, 0]*J[1, 1] + J[1, 0]*J[0, 1]
            V[s+4, s+5] = J[1, 0]*J[1, 1]
            V[s+5, s+3] = J[0, 1]*J[0, 1]
            V[s+5, s+4] = 2*J[0, 1]*J[1, 1]
            V[s+5, s+5] = J[1, 1]*J[1, 1]

        q = self.degree - 4
        if self.variant == "integral":
            _edge_transform(V, voffset, self.cell, q, coordinate_mapping, avg=self.avg)
        else:
            rns = coordinate_mapping.reference_normals()
            pns = coordinate_mapping.physical_normals()
            pts = coordinate_mapping.physical_tangents()
            pel = coordinate_mapping.physical_edge_lengths()
            for e in sorted(top[1]):
                nhat = partial_indexed(rns, (e, ))
                n = partial_indexed(pns, (e, ))
                t = partial_indexed(pts, (e, ))
                Bn = J @ nhat
                Bnt = Bn @ t
                Bnn = Bn @ n

                s = len(top[0]) * voffset + e
                v0id, v1id = (v * voffset for v in top[1][e])
                V[s, s] = Bnn

                # vertex points
                V[s, v0id] = -15/8 * Bnt / pel[e]
                V[s, v1id] = 15/8 * Bnt / pel[e]

                # vertex derivatives
                for i in range(sd):
                    V[s, v0id+1+i] = -7/16 * Bnt * t[i]
                    V[s, v1id+1+i] = V[s, v0id+1+i]

                # second derivatives
                tau = [t[0]*t[0], 2*t[0]*t[1], t[1]*t[1]]
                for i in range(len(tau)):
                    V[s, v0id+3+i] = -1/32 * (pel[e] * Bnt * tau[i])
                    V[s, v1id+3+i] = 1/32 * (pel[e] * Bnt * tau[i])

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            for k in range(sd):
                V[:, voffset*v+1+k] *= 1 / h[v]
            for k in range((sd+1)*sd//2):
                V[:, voffset*v+3+k] *= 1 / (h[v]*h[v])

        eoffset = 2 * q - 1
        for e in sorted(top[1]):
            v0, v1 = top[1][e]
            s = len(top[0]) * voffset + e * eoffset
            V[:, s:s+q] *= 2 / (h[v0] + h[v1])

        return ListTensor(V.T)
