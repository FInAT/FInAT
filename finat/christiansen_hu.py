import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from copy import deepcopy


class ChristiansenHu(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if degree != 2:
            raise ValueError("Christiansen-Hu only defined for degree == 2")
        if Citations is not None:
            Citations().register("AlfeldSorokina2016")
        super().__init__(FIAT.ChristiansenHu(cell, degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = reduced_dofs[sd-1][entity][:1]
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_dimension()
        top = self.cell.get_topology()
        # Jacobians at cell center
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        # Adjugate matrix
        rot = Literal(numpy.asarray([[0, 1], [-1, 0]]))
        K = -1 * rot @ J @ rot

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        edofs = self.entity_dofs()
        voffset = len(edofs[0][0])
        noffset = len(edofs[1][0])
        for v in sorted(top[0]):
            s = v * voffset
            for i in range(sd):
                for j in range(sd):
                    V[s+i, s+j] = K[j, i]

        rns = coordinate_mapping.reference_normals()
        pts = coordinate_mapping.physical_tangents()
        pel = coordinate_mapping.physical_edge_lengths()
        toffset = len(top[1]) * noffset

        ref_el = self.cell
        top = ref_el.get_topology()
        for e in sorted(top[1]):
            s = len(top[0]) * voffset + e * noffset
            nhat = partial_indexed(rns, (e, ))
            t = partial_indexed(pts, (e, ))
            Bnt = (J @ nhat) @ t
            rel = ref_el.volume_of_subcomplex(1, e)
            V[s+toffset, s] = Literal(-rel) * Bnt / pel[e]

        return ListTensor(V.T)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (9,)

    def space_dimension(self):
        return 9
