import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

from finat.argyris import _edge_transform
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from copy import deepcopy


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree, avg=False):
        if degree < 3:
            raise ValueError("HCT only defined for degree >= 3")
        if Citations is not None:
            Citations().register("Clough1965")
        self.avg = avg
        super().__init__(FIAT.HsiehCloughTocher(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        top = self.cell.get_topology()
        sd = self.cell.get_dimension()
        ndof = self.space_dimension()
        V = numpy.eye(ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        voffset = sd + 1
        num_verts = len(top[0])
        num_edges = len(top[1])
        for v in range(num_verts):
            s = voffset * v
            for i in range(sd):
                for j in range(sd):
                    V[s+1+i, s+1+j] = J[j, i]

        q = self.degree - 2
        _edge_transform(V, voffset, self.cell, q, coordinate_mapping, avg=self.avg)

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in range(num_verts):
            s = voffset * v
            for k in range(sd):
                V[:, s+1+k] /= h[v]

        eoffset = 2 * q - 1
        for e in range(num_edges):
            v0id, v1id = [i for i in range(num_verts) if i != e]
            s0 = voffset*num_verts + e * eoffset
            V[:, s0:s0+q] *= 2 / (h[v0id] + h[v1id])
        return ListTensor(V.T)


class ReducedHsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 3:
            raise ValueError("Degree must be 3 for reduced HCT element")
        if Citations is not None:
            Citations().register("Clough1965")
        super().__init__(FIAT.HsiehCloughTocher(cell, reduced=True))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        fdim = sd - 1
        for entity in reduced_dofs[fdim]:
            reduced_dofs[fdim][entity] = []
        self._entity_dofs = reduced_dofs

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
        return self._entity_dofs

    @property
    def index_shape(self):
        return (9,)

    def space_dimension(self):
        return 9
