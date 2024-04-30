import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 3:
            raise ValueError("Degree must be 3 for HCT element")
        if Citations is not None:
            Citations().register("Clough1965")
        super().__init__(FIAT.HsiehCloughTocher(cell))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
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

            Bn = nhat @ J.T / pel[e]
            Bnn = Bn @ n
            Bnt = (Bn @ t)

            V[s, s] = Bnn
            V[s, v0id] = Literal(-1) * Bnt
            V[s, v1id] = Bnt

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in range(d+1):
            s = voffset * v
            for k in range(d):
                V[:, s+1+k] /= h[v]
        for e in range(3):
            v0id, v1id = [i for i in range(3) if i != e]
            V[:, 9+e] *= 2 / (h[v0id] + h[v1id])
        return ListTensor(V.T)


class ReducedHsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 3:
            raise ValueError("Degree must be 3 for reduced HCT element")
        if Citations is not None:
            Citations().register("Clough1965")
        super().__init__(FIAT.HsiehCloughTocher(cell, reduced=True))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        rns = coordinate_mapping.reference_normals()
        rts = coordinate_mapping.normalized_reference_edge_tangents()
        pns = coordinate_mapping.physical_normals()
        pts = coordinate_mapping.physical_tangents()

        pel = coordinate_mapping.physical_edge_lengths()

        d = self.cell.get_dimension()

        # rectangular to toss out the constraint dofs
        V = numpy.eye(12, 12, dtype=object)
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

            Bnn = n @ J @ nhat
            Bnt = t @ J @ nhat

            V[s, s] = Bnn

            V[s, v0id] = Literal(0.2) * Bnt / pel[e]
            V[s, v0id + 1] = t[0] * Literal(0.1) * Bnt
            V[s, v0id + 2] = t[1] * Literal(0.1) * Bnt

            V[s, v1id] = Literal(-0.2) * Bnt / pel[e]
            V[s, v1id + 1] = t[0] * Literal(0.1) * Bnt
            V[s, v1id + 2] = t[1] * Literal(0.1) * Bnt

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in range(d+1):
            s = voffset * v
            for k in range(d):
                V[:, s+1+k] /= h[v]
        return ListTensor(V.T[:9])

    def entity_dofs(self):
        return {0: {0: list(range(3)),
                    1: list(range(3, 6)),
                    2: list(range(6, 9))},
                1: {0: [], 1: [], 2: []},
                2: {0: []}}

    @property
    def index_shape(self):
        return (9,)

    def space_dimension(self):
        return 9
