import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from copy import deepcopy


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree, avg=False):
        if degree != 3:
            raise ValueError("Degree must be 3 for HCT element")
        if Citations is not None:
            Citations().register("Clough1965")
        self.avg = avg
        super().__init__(FIAT.HsiehCloughTocher(cell))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        rns = coordinate_mapping.reference_normals()
        pts = coordinate_mapping.physical_tangents()
        pns = coordinate_mapping.physical_normals()

        pel = coordinate_mapping.physical_edge_lengths()

        d = self.cell.get_dimension()
        ndof = self.space_dimension()
        V = numpy.eye(ndof, dtype=object)
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

            nhat = partial_indexed(rns, (e, ))
            t = partial_indexed(pts, (e, ))
            n = partial_indexed(pns, (e, ))

            Bn = J @ nhat / pel[e]
            Bnn = Bn @ n
            Bnt = Bn @ t

            if self.avg:
                Bnn = Bnn * pel[e]
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
        pts = coordinate_mapping.physical_tangents()
        # pns = coordinate_mapping.physical_normals()

        pel = coordinate_mapping.physical_edge_lengths()

        d = self.cell.get_dimension()
        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
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

            nhat = partial_indexed(rns, (e, ))
            t = partial_indexed(pts, (e, ))

            # n = partial_indexed(pns, (e, ))
            # Bnn = (J @ nhat) @ n
            # V[s, s] = Bnn

            Bnt = (J @ nhat) @ t
            V[s, v0id] = Literal(1/5) * Bnt / pel[e]
            V[s, v1id] = Literal(-1) * V[s, v0id]

            R = Literal(1/10) * Bnt * t
            V[s, v0id + 1] = R[0]
            V[s, v0id + 2] = R[1]
            V[s, v1id + 1] = R[0]
            V[s, v1id + 2] = R[1]

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in range(d+1):
            s = voffset * v
            for k in range(d):
                V[:, s+1+k] /= h[v]
        return ListTensor(V.T)

    def entity_dofs(self):
        edofs = deepcopy(super(ReducedHsiehCloughTocher, self).entity_dofs())
        dim = 1
        for entity in edofs[dim]:
            edofs[dim][entity] = []
        return edofs

    @property
    def index_shape(self):
        return (9,)

    def space_dimension(self):
        return 9
