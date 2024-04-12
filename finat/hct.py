import numpy

import FIAT

from gem import Literal, ListTensor, partial_indexed

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
        pns = coordinate_mapping.physical_normals()
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
            for i in range(d):
                for j in range(d):
                    V[s+1+i, s+1+j] = J[j, i]

        for e in range(3):
            s = (d+1) * voffset + e
            v0id, v1id = [i * voffset for i in range(3) if i != e]

            t = partial_indexed(pts, (e, ))
            n = partial_indexed(pns, (e, ))
            nhat = partial_indexed(rns, (e, ))

            Bn = nhat @ J.T
            Bnn = Bn @ n
            Bnt = (Bn @ t) / pel[e]

            V[s, s] = Bnn
            V[s, v0id] = Literal(-1) * Bnt
            V[s, v1id] = Bnt

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in range(d+1):
            s = voffset * v
            for k in range(d):
                V[:, s+1+k] /= h[v]
        return ListTensor(V.T)
