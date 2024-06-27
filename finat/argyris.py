import numpy

import FIAT

from gem import Literal, ListTensor, partial_indexed

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


def _edge_transform(V, voffset, fiat_cell, moment_deg, coordinate_mapping, avg=False):
    """Basis transformation for integral edge moments."""

    A = numpy.zeros((moment_deg, moment_deg))
    if moment_deg > 1:
        A[1, 0] = -1.0
    for k in range(2, moment_deg):
        a, b, c = FIAT.expansions.jrc(0, 0, k-1)
        A[k, :k-2] -= (c/(1-a)) * A[k-2, :k-2]
        A[k, :k-1] += (b/(1-a)) * A[k-1, :k-1]
        A[k, k-1] = (k-1)*a/(1-a)

    J = coordinate_mapping.jacobian_at([1/3, 1/3])
    rns = coordinate_mapping.reference_normals()
    pts = coordinate_mapping.physical_tangents()
    pns = coordinate_mapping.physical_normals()
    pel = coordinate_mapping.physical_edge_lengths()

    top = fiat_cell.get_topology()
    num_verts = len(top[0])

    eoffset = 2 * moment_deg - 1
    for e in sorted(top[1]):
        v0id, v1id = [i * voffset for i in range(num_verts) if i != e]
        nhat = partial_indexed(rns, (e, ))
        n = partial_indexed(pns, (e, ))
        t = partial_indexed(pts, (e, ))
        Bn = J @ nhat / pel[e]
        Bnt = Bn @ t
        Bnn = Bn @ n
        if avg:
            Bnn = Bnn * pel[e]

        s0 = num_verts * voffset + e * eoffset
        toffset = s0 + moment_deg
        for k in range(moment_deg):
            s = s0 + k
            V[s, s] = Bnn
            V[s, v1id] = Bnt
            V[s, v0id] = Literal((-1)**(k+1)) * Bnt
            for j in range(k):
                V[s, toffset + j] = Literal(A[k, j]) * Bnt


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

        voffset = 6
        for v in range(3):
            s = voffset*v
            for i in range(2):
                for j in range(2):
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
            for e in range(3):
                v0id, v1id = [i*voffset for i in range(3) if i != e]
                s = 3*voffset + e
                nhat = partial_indexed(rns, (e, ))
                n = partial_indexed(pns, (e, ))
                t = partial_indexed(pts, (e, ))
                Bn = J @ nhat
                Bnt = Bn @ t
                Bnn = Bn @ n
                V[s, s] = Bnn

                # vertex points
                V[s, v0id] = -15/8 * Bnt / pel[e]
                V[s, v1id] = 15/8 * Bnt / pel[e]

                # vertex derivatives
                for i in range(2):
                    V[s, v0id+1+i] = -7/16 * Bnt * t[i]
                    V[s, v1id+1+i] = V[s, v0id+1+i]

                # second derivatives
                tau = [t[0]*t[0], 2*t[0]*t[1], t[1]*t[1]]
                for i in range(3):
                    V[s, v0id+3+i] = -1/32 * (pel[e] * Bnt * tau[i])
                    V[s, v1id+3+i] = 1/32 * (pel[e] * Bnt * tau[i])

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in range(3):
            for k in range(2):
                V[:, voffset*v+1+k] *= 1 / h[v]
            for k in range(3):
                V[:, voffset*v+3+k] *= 1 / (h[v]*h[v])

        eoffset = 2 * q - 1
        for e in range(3):
            s = voffset*3 + e*eoffset
            v0id, v1id = [i for i in range(3) if i != e]
            V[:, s:s+q] *= 2 / (h[v0id] + h[v1id])

        return ListTensor(V.T)
