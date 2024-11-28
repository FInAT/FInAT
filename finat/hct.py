import FIAT
from math import comb
from gem import ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement
from finat.argyris import _vertex_transform, _edge_transform, _normal_tangential_transform
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
        V = identity(self.space_dimension())

        sd = self.cell.get_dimension()
        top = self.cell.get_topology()

        vorder = 1
        eorder = self.degree - 3
        voffset = comb(sd + vorder, vorder)
        _vertex_transform(V, vorder, self.cell, coordinate_mapping)
        _edge_transform(V, vorder, eorder, self.cell, coordinate_mapping, avg=self.avg)

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            s = voffset*v + 1
            V[:, s:s+sd] *= 1/h[v]
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
        sd = self.cell.get_spatial_dimension()
        top = self.cell.get_topology()
        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = identity(numbf, ndof)

        vorder = 1
        voffset = comb(sd + vorder, vorder)
        _vertex_transform(V, vorder, self.cell, coordinate_mapping)

        # Jacobian at barycenter
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        for e in sorted(top[1]):
            s = len(top[0]) * voffset + e
            v0id, v1id = (v * voffset for v in top[1][e])
            Bnn, Bnt, Jt = _normal_tangential_transform(self.cell, J, detJ, e)

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
            s = voffset * v + 1
            V[:, s:s+sd] *= 1/h[v]
        return ListTensor(V.T)

    # This wipes out the edge dofs.  FIAT gives a 12 DOF element
    # because we need some extra functions to help with transforming
    # under the edge constraint.  However, we only have a 9 DOF
    # element.
    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (9,)

    def space_dimension(self):
        return 9
