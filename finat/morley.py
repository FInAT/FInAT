import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class Morley(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2):
        if degree != 2:
            raise ValueError("Degree must be 2 for Morley element")
        if Citations is not None:
            Citations().register("Morley1971")
        super().__init__(FIAT.Morley(cell))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at edge midpoints
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        rns = coordinate_mapping.reference_normals()
        pns = coordinate_mapping.physical_normals()

        pts = coordinate_mapping.physical_tangents()

        pel = coordinate_mapping.physical_edge_lengths()

        V = numpy.eye(6, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        for i in range(3):
            V[i+3, i+3] = (rns[i, 0]*(pns[i, 0]*J[0, 0] + pns[i, 1]*J[1, 0])
                           + rns[i, 1]*(pns[i, 0]*J[0, 1] + pns[i, 1]*J[1, 1]))

        for i, c in enumerate([(1, 2), (0, 2), (0, 1)]):
            B12 = (rns[i, 0]*(pts[i, 0]*J[0, 0] + pts[i, 1]*J[1, 0])
                   + rns[i, 1]*(pts[i, 0]*J[0, 1] + pts[i, 1]*J[1, 1]))
            V[3+i, c[0]] = -1*B12 / pel[i]
            V[3+i, c[1]] = B12 / pel[i]

        # diagonal post-scaling to patch up conditioning
        h = coordinate_mapping.cell_size()

        for e in range(3):
            v0id, v1id = [i for i in range(3) if i != e]
            for i in range(6):
                V[i, 3+e] = 2*V[i, 3+e] / (h[v0id] + h[v1id])

        return ListTensor(V.T)
