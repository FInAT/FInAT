import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


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


class AlfeldSorokina(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if degree != 2:
            raise ValueError("Alfeld-Sorokina only defined for degree = 2")
        if Citations is not None:
            Citations().register("AlfeldSorokina2016")
        super().__init__(FIAT.AlfeldSorokina(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        cofJ = cofactor([[J[j, i] for j in range(sd)] for i in range(sd)])

        ndof = self.space_dimension()
        V = numpy.eye(ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        edofs = self.entity_dofs()
        cur = 0
        for dim in edofs:
            for entity in sorted(edofs[dim]):
                s = cur
                cur += len(edofs[dim][entity])
                if dim == 0:
                    V[s, s] = detJ
                    s += 1
                for i in range(s, cur, sd):
                    V[i:i+sd, i:i+sd] = cofJ

        return ListTensor(V.T)
