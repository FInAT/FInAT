import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from copy import deepcopy


def determinant(A):
    n = A.shape[0]
    if n == 1:
        return A[0, 0]
    elif n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    else:
        raise NotImplementedError(f"{n}-by-{n} determinants not implemented")


def cofactor(A):
    """Return the cofactor matrix of A"""
    A = numpy.asarray(A)
    sel_rows = numpy.ones(A.shape[0], dtype=bool)
    sel_cols = numpy.ones(A.shape[1], dtype=bool)
    C = numpy.zeros_like(A)
    sgn_row = 1
    for row in range(A.shape[0]):
        sel_rows[row] = False
        sgn_col = 1
        for col in range(A.shape[1]):
            sel_cols[col] = False
            C[row, col] = sgn_row*sgn_col*determinant(A[sel_rows, :][:, sel_cols])
            sel_cols[col] = True
            sgn_col = -sgn_col
        sel_rows[row] = True
        sgn_row = -sgn_row
    return C


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
        cofJ = cofactor([[J[j, i] for j in range(sd)] for i in range(sd)])

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        edofs = self.entity_dofs()
        for v in sorted(edofs[0]):
            vdofs = edofs[0][v]
            V[numpy.ix_(vdofs, vdofs)] = cofJ

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
        return ListTensor(V.T)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    def space_dimension(self):
        return (self.cell.get_spatial_dimension() + 1)**2
