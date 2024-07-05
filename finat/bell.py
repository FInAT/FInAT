import numpy

import FIAT

from gem import Literal, ListTensor, partial_indexed

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class Bell(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 5:
            raise ValueError("Degree must be 3 for Bell element")
        if Citations is not None:
            Citations().register("Bell1969")
        super().__init__(FIAT.Bell(cell))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at edge midpoints
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        voffset = sd + 1 + (sd*(sd+1))//2
        for v in sorted(top[1]):
            s = voffset * v
            V[s, s] = Literal(1)
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

        rns = coordinate_mapping.reference_normals()
        pts = coordinate_mapping.physical_tangents()
        pel = coordinate_mapping.physical_edge_lengths()
        for e in sorted(top[1]):
            s = len(top[0]) * voffset + e
            v0id, v1id = (v * voffset for v in top[1][e])

            nhat = partial_indexed(rns, (e, ))
            t = partial_indexed(pts, (e, ))
            Bnt = (J @ nhat) @ t

            # vertex points
            V[s, v1id] = 1/21 * Bnt / pel[e]
            V[s, v0id] = -1 * V[s, v1id]

            # vertex derivatives
            for i in range(sd):
                V[s, v1id+1+i] = -1/42 * Bnt * t[i]
                V[s, v0id+1+i] = V[s, v1id+1+i]

            # second derivatives
            tau = [t[0]*t[0], 2*t[0]*t[1], t[1]*t[1]]
            for i in range(len(tau)):
                V[s, v1id+3+i] = 1/252 * (pel[e] * Bnt * tau[i])
                V[s, v0id+3+i] = -1 * V[s, v1id+3+i]

        h = coordinate_mapping.cell_size()

        for v in sorted(top[0]):
            for k in range(sd):
                V[:, voffset*v+1+k] *= 1/h[v]
            for k in range((sd+1)*sd//2):
                V[:, voffset*v+3+k] *= 1/(h[v]*h[v])

        return ListTensor(V.T)

    # This wipes out the edge dofs.  FIAT gives a 21 DOF element
    # because we need some extra functions to help with transforming
    # under the edge constraint.  However, we only have an 18 DOF
    # element.
    def entity_dofs(self):
        return {0: {0: list(range(6)),
                    1: list(range(6, 12)),
                    2: list(range(12, 18))},
                1: {0: [], 1: [], 2: []},
                2: {0: []}}

    @property
    def index_shape(self):
        return (18,)

    def space_dimension(self):
        return 18
