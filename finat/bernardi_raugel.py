import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from finat.piola_mapped import piola_inverse, normal_tangential_edge_transform, normal_tangential_face_transform
from copy import deepcopy


def bernardi_raugel_transformation(self, coordinate_mapping):
    sd = self.cell.get_spatial_dimension()
    bary, = self.cell.make_points(sd, 0, sd+1)
    J = coordinate_mapping.jacobian_at(bary)
    detJ = coordinate_mapping.detJ_at(bary)

    dofs = self.entity_dofs()
    edofs = self._element.entity_dofs()
    ndof = self.space_dimension()
    numbf = self._element.space_dimension()
    V = numpy.eye(numbf, ndof, dtype=object)
    for multiindex in numpy.ndindex(V.shape):
        V[multiindex] = Literal(V[multiindex])

    Finv = piola_inverse(self.cell, J, detJ)
    for v in sorted(dofs[0]):
        vdofs = dofs[0][v]
        if len(vdofs) == sd:
            V[numpy.ix_(vdofs, vdofs)] = Finv

    if sd == 2:
        transform = normal_tangential_edge_transform
    elif sd == 3:
        transform = normal_tangential_face_transform

    for f in sorted(dofs[sd-1]):
        rows = numpy.asarray(transform(self.cell, J, detJ, f))
        fdofs = dofs[sd-1][f]
        bfs = edofs[sd-1][f][1:]
        V[numpy.ix_(bfs, fdofs)] = rows[..., :len(fdofs)]
    return ListTensor(V.T)


class BernardiRaugel(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=None):
        sd = cell.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError(f"Bernardi-Raugel only defined for degree = {sd}")
        if Citations is not None:
            Citations().register("BernardiRaugel1985")
        super().__init__(FIAT.BernardiRaugel(cell, degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        cur = reduced_dofs[sd-1][0][0]
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = [cur]
            cur += 1
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        return bernardi_raugel_transformation(self, coordinate_mapping)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    def space_dimension(self):
        return (self.cell.get_spatial_dimension() + 1)**2
