"""Implementation of the Arnold-Winther finite elements."""
import numpy
import FIAT
from gem import Literal, ListTensor, partial_indexed
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


def _face_transform(coordinate_mapping):
    sd = coordinate_mapping.ref_cell.get_spatial_dimension()
    num_faces = sd + 1
    dimP1_face = sd
    dofs_per_face = sd * dimP1_face
    ndofs = num_faces * dofs_per_face

    Vsub = numpy.eye(ndofs, dtype=object)
    for multiindex in numpy.ndindex(Vsub.shape):
        Vsub[multiindex] = Literal(Vsub[multiindex])

    bary = [1/(sd+1)] * sd
    detJ = coordinate_mapping.detJ_at(bary)
    J = coordinate_mapping.jacobian_at(bary)
    R = Literal(numpy.array([[0, -1], [1, 0]]))
    rns = coordinate_mapping.reference_normals()
    offset = dofs_per_face
    for e in range(num_faces):
        nhat = partial_indexed(rns, (e, ))
        that = R @ nhat
        Jn = J @ nhat
        Jt = J @ that

        # Compute alpha and beta for the edge.
        alpha = (Jn @ Jt) / detJ
        beta = (Jt @ Jt) / detJ
        # Stuff into the right rows and columns.
        idx1, idx2 = offset*e + 1, offset*(e+1) - 1
        Vsub[idx1, idx1-1] = Literal(-1) * alpha / beta
        Vsub[idx1, idx1] = Literal(1) / beta
        Vsub[idx2, idx2-1] = Literal(-1) * alpha / beta
        Vsub[idx2, idx2] = Literal(1) / beta

    return Vsub


def _evaluation_transform(coordinate_mapping):
    J = coordinate_mapping.jacobian_at([1/3, 1/3])

    W = numpy.zeros((3, 3), dtype=object)
    W[0, 0] = J[1, 1]*J[1, 1]
    W[0, 1] = -2*J[1, 1]*J[0, 1]
    W[0, 2] = J[0, 1]*J[0, 1]
    W[1, 0] = -1*J[1, 1]*J[1, 0]
    W[1, 1] = J[1, 1]*J[0, 0] + J[0, 1]*J[1, 0]
    W[1, 2] = -1*J[0, 1]*J[0, 0]
    W[2, 0] = J[1, 0]*J[1, 0]
    W[2, 1] = -2*J[1, 0]*J[0, 0]
    W[2, 2] = J[0, 0]*J[0, 0]

    return W


class ArnoldWintherNC(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        if Citations is not None:
            Citations().register("Arnold2003")
        super(ArnoldWintherNC, self).__init__(FIAT.ArnoldWintherNC(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """Note, the extra 3 dofs which are removed here
        correspond to the constraints."""
        V = numpy.zeros((18, 15), dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        V[:12, :12] = _face_transform(coordinate_mapping)

        # internal dofs
        W = _evaluation_transform(coordinate_mapping)
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])

        V[12:15, 12:15] = W / detJ

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.

        return ListTensor(V.T)

    def entity_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]},
                2: {0: [12, 13, 14]}}

    def entity_closure_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]},
                2: {0: list(range(15))}}

    @property
    def index_shape(self):
        return (15,)

    def space_dimension(self):
        return 15


class ArnoldWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        if Citations is not None:
            Citations().register("Arnold2002")
        super(ArnoldWinther, self).__init__(FIAT.ArnoldWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """The extra 6 dofs removed here correspond to the constraints."""
        V = numpy.zeros((30, 24), dtype=object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        W = _evaluation_transform(coordinate_mapping)

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W

        V[9:21, 9:21] = _face_transform(coordinate_mapping)

        # internal DOFs
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])
        V[21:24, 21:24] = W / detJ

        # RESCALING FOR CONDITIONING
        h = coordinate_mapping.cell_size()

        for e in range(3):
            eff_h = h[e]
            for i in range(24):
                V[i, 3*e] = V[i, 3*e]/(eff_h*eff_h)
                V[i, 1+3*e] = V[i, 1+3*e]/(eff_h*eff_h)
                V[i, 2+3*e] = V[i, 2+3*e]/(eff_h*eff_h)

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.

        return ListTensor(V.T)

    def entity_dofs(self):
        return {0: {0: [0, 1, 2],
                    1: [3, 4, 5],
                    2: [6, 7, 8]},
                1: {0: [9, 10, 11, 12], 1: [13, 14, 15, 16], 2: [17, 18, 19, 20]},
                2: {0: [21, 22, 23]}}

    # need to overload since we're cutting out some dofs from the FIAT element.
    def entity_closure_dofs(self):
        ct = self.cell.topology
        ecdofs = {i: {} for i in range(3)}
        for i in range(3):
            ecdofs[0][i] = list(range(3*i, 3*(i+1)))

        for i in range(3):
            ecdofs[1][i] = [dof for v in ct[1][i] for dof in ecdofs[0][v]] + \
                list(range(9+4*i, 9+4*(i+1)))

        ecdofs[2][0] = list(range(24))

        return ecdofs

    @property
    def index_shape(self):
        return (24,)

    def space_dimension(self):
        return 24
