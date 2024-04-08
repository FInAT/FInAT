import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 3:
            raise ValueError("Degree must be 3 for HCT element")
        if Citations is not None:
            Citations().register("Clough1965")
        super().__init__(FIAT.HsiehCloughTocher(cell))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at edge midpoints
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        rns = coordinate_mapping.reference_normals()

        pts = coordinate_mapping.physical_tangents()

        pel = coordinate_mapping.physical_edge_lengths()

        d = self.cell.get_dimension()
        numbf = self.space_dimension()

        V = numpy.eye(numbf, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        voffset = d+1
        for v in range(d+1):
            s = voffset * v
            V[s, s] = Literal(1)
            for i in range(d):
                for j in range(d):
                    V[s+1+j, s+1+i] = J[j, i]

        for e in range(3):
            s = 3 * voffset + e
            v0id, v1id = [i * voffset for i in range(3) if i != e]

            # nhat . J^{-T} . t
            foo = (rns[e, 0]*(J[0, 0]*pts[e, 0] + J[1, 0]*pts[e, 1])
                   + rns[e, 1]*(J[0, 1]*pts[e, 0] + J[1, 1]*pts[e, 1]))

            # vertex points
            V[v0id, s] = -1/21 * (foo / pel[e])
            V[v1id, s] = 1/21 * (foo / pel[e])

            # vertex derivatives
            for i in range(d):
                V[v0id+1+i, s] = -1/42*foo*pts[e, i]
                V[v1id+1+i, s] = V[v0id+1+i, s]

        h = coordinate_mapping.cell_size()

        for v in range(d+1):
            s = voffset * v
            for k in range(d):
                V[s+1+k, :] /= h[v]

        return ListTensor(V)
