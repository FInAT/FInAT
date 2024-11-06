import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class Bell(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
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
            V[s, v1id] = 1/21 * Bnt
            V[s, v0id] = -1 * V[s, v1id]

            # vertex derivatives
            for i in range(sd):
                V[s, v1id+1+i] = -1/42 * Bnt * Jt[i]
                V[s, v0id+1+i] = V[s, v1id+1+i]

            # second derivatives
            tau = [Jt[0]*Jt[0], 2*Jt[0]*Jt[1], Jt[1]*Jt[1]]
            for i in range(len(tau)):
                V[s, v1id+3+i] = 1/252 * (Bnt * tau[i])
                V[s, v0id+3+i] = -1 * V[s, v1id+3+i]

        # Patch up conditioning
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
