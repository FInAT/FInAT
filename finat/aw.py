"""Implementation of the Arnold-Winther finite elements."""
import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class ArnoldWintherNC(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        super(ArnoldWintherNC, self).__init__(FIAT.ArnoldWintherNC(cell, degree))

    @staticmethod
    def basis_transformation(self, coordinate_mapping, as_numpy=False):
        """Note, the extra 3 dofs which are removed here
        correspond to the constraints."""
        V = numpy.zeros((18, 15), dtype=object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        for i in range(0, 12, 2):
            V[i, i] = Literal(1)

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

            (alpha, beta) = Ghat_T @ JTJ @ that[e,:] / detJ

            # Stuff into the right rows and columns.
            (idx1, idx2) = (4*e + 1, 4*e + 3)
            V[idx1, idx1-1] = Literal(-1) * alpha / beta
            V[idx1, idx1] = Literal(1) / beta
            V[idx2, idx2-1] = Literal(-1) * alpha / beta
            V[idx2, idx2] = Literal(1) / beta

        # internal dofs
        for i in range(12, 15):
            V[i, i] = Literal(1)

        if as_numpy: 
            return V.T
        else:
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

class ArnoldWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        super(ArnoldWinther, self).__init__(FIAT.ArnoldWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """The extra 6 dofs removed here correspond to
        the constraints."""
        V = numpy.zeros((30, 24), dtype=object)

        # The edge and internal dofs are as for the
        # nonconforming element.
        V[9:24, 9:24] = (ArnoldWintherNC.basis_transformation(self, coordinate_mapping, True)).T

        # vertex dofs
        # TODO: find a succinct expression for W in terms of J.
        J = coordinate_mapping.jacobian_at([1/3, 1/3])
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])

        W = numpy.zeros((3,3), dtype=object)
        W[0, 0] = J[0, 0]*J[0, 0]
        W[0, 1] = 2*J[0, 0]*J[0, 1]
        W[0, 2] = J[0, 1]*J[0, 1]
        W[1, 0] = J[0, 0]*J[1, 0]
        W[1, 1] = J[0, 0]*J[1, 1] + J[0, 1]*J[1, 0]
        W[1, 2] = J[0, 1]*J[1, 1]
        W[2, 0] = J[1, 0]*J[1, 0]
        W[2, 1] = 2*J[1, 0]*J[1, 1]
        W[2, 2] = J[1, 1]*J[1, 1]
        W = W / detJ

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W

        return ListTensor(V.T)


    def entity_dofs(self):
        return {0: {0: [0, 1, 2],
                    1: [3, 4, 5],
                    2: [6, 7, 8]},
                1: {0: [9, 10, 11, 12], 1: [13, 14, 15, 16], 2: [17, 18, 19, 20]},
                2: {0: [21, 22, 23]}}


    @property
    def index_shape(self):
        return (24,)


    def space_dimension(self):
        return 24
