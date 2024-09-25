import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement, adjugate
from copy import deepcopy


class BernardiRaugel(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=None):
        sd = cell.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError(f"Bernardi-Raugel only defined for degree = {sd}")
        if Citations is not None:
            Citations().register("ArnoldQin1992")
        super().__init__(FIAT.BernardiRaugel(cell, degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = reduced_dofs[sd-1][entity][:1]
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        adjJ = adjugate([[J[i, j] for j in range(sd)] for i in range(sd)])

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        edofs = self.entity_dofs()
        for v in sorted(edofs[0]):
            vdofs = edofs[0][v]
            V[numpy.ix_(vdofs, vdofs)] = adjJ

        rns = coordinate_mapping.reference_normals()
        pts = coordinate_mapping.physical_tangents()
        pel = coordinate_mapping.physical_edge_lengths()
        toffset = len(edofs[sd-1])

        for e in sorted(edofs[sd-1]):
            sn = edofs[sd-1][e][0]
            st = sn + toffset
            nhat = partial_indexed(rns, (e, ))
            t = partial_indexed(pts, (e, ))
            Bnt = (J @ nhat) @ t
            rel = self.cell.volume_of_subcomplex(sd-1, e)
            scale = rel / pel[e]
            V[st, sn] = -1 * scale * Bnt
        return ListTensor(V.T)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    def space_dimension(self):
        return (self.cell.get_spatial_dimension() + 1)**2
