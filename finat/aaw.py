import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class ArnoldAwanouWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        super(ArnoldAwanouWinther, self).__init__(FIAT.ArnoldAwanouWinther(cell, degree))


    def basis_transformation(self, coordinate_mapping):
        V = numpy.zeros((18, 15), dtype=object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        for i in range(0, 12, 2):
            V[i, i] = Literal(1)


        detJ = coordinate_mapping.detJ_at([1/3, 1/3])
        J = coordinate_mapping.jacobian_at([1/3, 1/3])
        J_np = numpy.array([[J[0, 0], J[0, 1]],
                            [J[1, 0], J[1, 1]]])
        JTJ = J_np.T @ J_np
        rts = coordinate_mapping.reference_edge_tangents()
        rns = coordinate_mapping.reference_normals()

        for e in range(3):
            # update rows 1,3 for edge 0,
            #        rows 5,7 for edge 1,
            #        rows 9, 11 for edge 2.

            # Compute alpha and beta for the edge.
            Ghat = numpy.array([[rns[e, 0], rts[e, 0]],
                                [rns[e, 1], rts[e, 1]]])
            that = numpy.array([rts[e, 0], rts[e, 1]])
            (alpha, beta) = Ghat @ JTJ @ that / detJ

            # Stuff into the right rows and columns.
            (idx1, idx2) = (4*e + 1, 4*e + 3)
            V[idx1, idx1-1] = Literal(-1) * alpha / beta
            V[idx1, idx1] = Literal(1) / beta
            V[idx2, idx2-1] = Literal(-1) * alpha / beta
            V[idx2, idx2] = Literal(1) / beta

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


    @property
    def index_shape(self):
        return (15,)


    def space_dimension(self):
        return 15
