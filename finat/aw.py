"""Implementation of the Arnold-Winther finite elements."""
import numpy

import FIAT

from gem import Literal, ListTensor, partial_indexed

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


def _edge_transform(T, coordinate_mapping):
    Vsub = numpy.zeros((12, 12), dtype=object)

    for multiindex in numpy.ndindex(Vsub.shape):
        Vsub[multiindex] = Literal(Vsub[multiindex])

    for i in range(0, 12, 2):
        Vsub[i, i] = Literal(1)

    nhat = coordinate_mapping.reference_normals()
    that = coordinate_mapping.reference_edge_tangents(normalized=True)

    detJ = coordinate_mapping.detJ_at([1/3, 1/3])
    J = coordinate_mapping.jacobian_at([1/3, 1/3])

    JTJ = J.T @ J

    for e in range(3):
        nhat_cur = partial_indexed(nhat, (e,))
        that_cur = partial_indexed(that, (e,))
        blah = JTJ @ that_cur / detJ
        alpha = nhat_cur @ blah
        beta = that_cur @ blah

        # Stuff into the right rows and columns.
        (idx1, idx2) = (4*e + 1, 4*e + 3)
        Vsub[idx1, idx1-1] = Literal(-1) * alpha / beta
        Vsub[idx1, idx1] = Literal(1) / beta
        Vsub[idx2, idx2-1] = Literal(-1) * alpha / beta
        Vsub[idx2, idx2] = Literal(1) / beta

    return Vsub


class ArnoldWintherNC(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        if Citations is not None:
            Citations().register("Arnold2003")
        super(ArnoldWintherNC, self).__init__(FIAT.ArnoldWintherNC(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """Note, the extra 3 dofs which are removed here
        correspond to the constraints."""

        T = self.cell
        V = numpy.zeros((18, 15), dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        V[:12, :12] = _edge_transform(T, coordinate_mapping)

        # internal dofs
        for i in range(12, 15):
            V[i, i] = Literal(1)

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

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W

        V[9:21, 9:21] = _edge_transform(self.cell, coordinate_mapping)

        for i in range(21, 24):
            V[i, i] = Literal(1)

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
