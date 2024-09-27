import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement, adjugate
from finat.aw import _single_edge_transform, _single_face_transform
from copy import deepcopy


class ArnoldQin(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=None):
        if Citations is not None:
            Citations().register("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        adjJ = adjugate([[J[i, j] for j in range(sd)] for i in range(sd)])

        ndof = self.space_dimension()
        numbf = self._element.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        dofs = self.entity_dofs()
        edofs = self._element.entity_dofs()
        for v in sorted(dofs[0]):
            vdofs = dofs[0][v]
            V[numpy.ix_(vdofs, vdofs)] = adjJ

        fiat_cell = self.cell
        if sd == 2:
            transform = _single_edge_transform
        elif sd == 3:
            transform = _single_face_transform

        for f in sorted(dofs[sd-1]):
            rows = numpy.asarray(transform(fiat_cell, J, detJ, f))
            fdofs = dofs[sd-1][f]
            bfs = edofs[sd-1][f][1:]
            V[numpy.ix_(bfs, fdofs)] = rows[..., :len(fdofs)]
        return ListTensor(V.T)


class ReducedArnoldQin(ArnoldQin):
    def __init__(self, cell, degree=None):
        super(ReducedArnoldQin, self).__init__(cell, degree)

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = reduced_dofs[sd-1][entity][:1]
        self._entity_dofs = reduced_dofs

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    def space_dimension(self):
        return (self.cell.get_spatial_dimension() + 1)**2
