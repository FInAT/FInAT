import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from copy import deepcopy


class ArnoldQin(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if degree != 2:
            raise ValueError("Arnold-Qin only defined for degree = 2")
        if Citations is not None:
            Citations().register("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        # Adjugate matrix
        rot = Literal([[0, 1], [-1, 0]])
        K = -1 * rot @ J @ rot
        KT = [[K[j, i] for j in range(sd)] for i in range(sd)]

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        edofs = self.entity_dofs()
        for v in sorted(edofs[0]):
            vdofs = edofs[0][v]
            V[numpy.ix_(vdofs, vdofs)] = KT

        rns = coordinate_mapping.reference_normals()
        pts = coordinate_mapping.physical_tangents()
        pel = coordinate_mapping.physical_edge_lengths()
        voffset = len(edofs[0]) * len(edofs[0][0])
        toffset = len(edofs[sd-1])

        for e in sorted(edofs[sd-1]):
            sn = voffset + e
            st = sn + toffset
            nhat = partial_indexed(rns, (e, ))
            t = partial_indexed(pts, (e, ))
            Bnt = (J @ nhat) @ t
            rel = self.cell.volume_of_subcomplex(sd-1, e)
            scale = rel / pel[e]
            V[st, sn] = -1 * scale * Bnt
            if numbf == ndof:
                V[st, st] = scale * scale * detJ
        return ListTensor(V.T)


class ReducedArnoldQin(ArnoldQin):
    def __init__(self, cell, degree=2):
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
        return (9,)

    def space_dimension(self):
        return 9
