import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class Argyris(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 5:
            raise ValueError("Degree must be 5 for Argyris element")
        if Citations is not None:
            Citations().register("Argyris1968")
        super().__init__(FIAT.QuinticArgyris(cell))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at edge midpoints
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        rns = coordinate_mapping.reference_normals()
        pns = coordinate_mapping.physical_normals()

        pts = coordinate_mapping.physical_tangents()

        pel = coordinate_mapping.physical_edge_lengths()

        V = numpy.zeros((21, 21), dtype=object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        for v in range(3):
            s = 6*v
            V[s, s] = Literal(1)
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

        for e in range(3):
            v0id, v1id = [i for i in range(3) if i != e]

            # nhat . J^{-T} . t
            foo = (rns[e, 0]*(J[0, 0]*pts[e, 0] + J[1, 0]*pts[e, 1])
                   + rns[e, 1]*(J[0, 1]*pts[e, 0] + J[1, 1]*pts[e, 1]))

            # vertex points
            V[18+e, 6*v0id] = -15/8 * (foo / pel[e])
            V[18+e, 6*v1id] = 15/8 * (foo / pel[e])

            # vertex derivatives
            for i in (0, 1):
                V[18+e, 6*v0id+1+i] = -7/16*foo*pts[e, i]
                V[18+e, 6*v1id+1+i] = V[18+e, 6*v0id+1+i]

            # second derivatives
            tau = [pts[e, 0]*pts[e, 0],
                   2*pts[e, 0]*pts[e, 1],
                   pts[e, 1]*pts[e, 1]]

            for i in (0, 1, 2):
                V[18+e, 6*v0id+3+i] = -1/32 * (pel[e]*foo*tau[i])
                V[18+e, 6*v1id+3+i] = 1/32 * (pel[e]*foo*tau[i])

            V[18+e, 18+e] = (rns[e, 0]*(J[0, 0]*pns[e, 0] + J[1, 0]*pns[e, 1])
                             + rns[e, 1]*(J[0, 1]*pns[e, 0] + J[1, 1]*pns[e, 1]))

        # Patch up conditioning
        h = coordinate_mapping.cell_size()

        for v in range(3):
            for k in range(2):
                for i in range(21):
                    V[i, 6*v+1+k] = V[i, 6*v+1+k] / h[v]
            for k in range(3):
                for i in range(21):
                    V[i, 6*v+3+k] = V[i, 6*v+3+k] / (h[v]*h[v])
        for e in range(3):
            v0id, v1id = [i for i in range(3) if i != e]
            for i in range(21):
                V[i, 18+e] = 2*V[i, 18+e] / (h[v0id] + h[v1id])

        return ListTensor(V.T)
