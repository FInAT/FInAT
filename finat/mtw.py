import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3):
        if Citations is not None:
            Citations().register("Mardal2002")
        super(MardalTaiWinther, self).__init__(FIAT.MardalTaiWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        V = numpy.zeros((20, 9), dtype=object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        for i in range(0, 9, 3):
            V[i, i] = Literal(1)
            V[i+2, i+2] = Literal(1)

        T = self.cell

        # This bypasses the GEM wrapper.
        that = numpy.array([T.compute_normalized_edge_tangent(i) for i in range(3)])
        nhat = numpy.array([T.compute_normal(i) for i in range(3)])

        detJ = coordinate_mapping.detJ_at([1/3, 1/3])
        J = coordinate_mapping.jacobian_at([1/3, 1/3])
        J_np = numpy.array([[J[0, 0], J[0, 1]],
                            [J[1, 0], J[1, 1]]])
        JTJ = J_np.T @ J_np

        for e in range(3):
            # Compute alpha and beta for the edge.
            Ghat_T = numpy.array([nhat[e, :], that[e, :]])

            (alpha, beta) = Ghat_T @ JTJ @ that[e, :] / detJ

            # Stuff into the right rows and columns.
            idx = 3*e + 1
            V[idx, idx-1] = Literal(-1) * alpha / beta
            V[idx, idx] = Literal(1) / beta

        return ListTensor(V.T)

    def entity_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
                2: {0: []}}

    def entity_closure_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
                2: {0: [0, 1, 2, 3, 4, 5, 6, 7, 8]}}

    @property
    def index_shape(self):
        return (9,)

    def space_dimension(self):
        return 9
