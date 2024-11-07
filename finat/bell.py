import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations
from finat.argyris import _vertex_transform, _normal_tangential_transform
from copy import deepcopy


class Bell(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=5):
        if Citations is not None:
            Citations().register("Bell1969")
        super().__init__(FIAT.Bell(cell))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = []
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        # Jacobian at barycenter
        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        _vertex_transform(V, self.cell, J)

        voffset = sd + 1 + (sd*(sd+1))//2
        for e in sorted(top[1]):
            s = len(top[0]) * voffset + e
            v0id, v1id = (v * voffset for v in top[1][e])
            Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e)

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
        return self._entity_dofs

    @property
    def index_shape(self):
        return (18,)

    def space_dimension(self):
        return 18
