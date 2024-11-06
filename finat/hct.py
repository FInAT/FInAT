import FIAT
import numpy
from gem import ListTensor, Literal

from finat.argyris import _edge_transform
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from copy import deepcopy


class HsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3, avg=False):
        if Citations is not None:
            Citations().register("Clough1965")
            if degree > 3:
                Citations().register("Groselj2022")
        self.avg = avg
        super().__init__(FIAT.HsiehCloughTocher(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        ndof = self.space_dimension()
        V = numpy.eye(ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        sd = self.cell.get_dimension()
        top = self.cell.get_topology()
        voffset = 1 + sd
        for v in sorted(top[0]):
            s = voffset * v
            for i in range(sd):
                for j in range(sd):
                    V[s+1+i, s+1+j] = J[j, i]

        vorder = 1
        eorder = self.degree - 3
        _edge_transform(V, vorder, eorder, self.cell, coordinate_mapping, avg=self.avg)

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            for k in range(sd):
                V[:, voffset*v+1+k] /= h[v]
        return ListTensor(V.T)


class ReducedHsiehCloughTocher(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3):
        if Citations is not None:
            Citations().register("Clough1965")
        super().__init__(FIAT.HsiehCloughTocher(cell, reduced=True))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = []
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        # Jacobian at barycenter
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        voffset = sd + 1
        for v in sorted(top[0]):
            s = voffset * v
            for i in range(sd):
                for j in range(sd):
                    V[s+1+i, s+1+j] = J[j, i]

        for e in sorted(top[1]):
            s = len(top[0]) * voffset + e
            v0id, v1id = (v * voffset for v in top[1][e])

            that = self.cell.compute_edge_tangent(e)
            nhat = self.cell.compute_scaled_normal(e)
            nhat /= numpy.linalg.norm(nhat)
            Jt = J @ Literal(that)
            Jn = J @ Literal(nhat)
            Bnt = (Jn @ Jt) / (Jt @ Jt)

            # vertex points
            V[s, v0id] = 1/5 * Bnt
            V[s, v1id] = -1 * V[s, v0id]

            # vertex derivatives
            for i in range(sd):
                V[s, v1id+1+i] = 1/10 * Bnt * Jt[i]
                V[s, v0id+1+i] = V[s, v1id+1+i]

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            s = voffset * v
            for k in range(sd):
                V[:, s+1+k] /= h[v]
        return ListTensor(V.T)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (9,)

    def space_dimension(self):
        return 9
