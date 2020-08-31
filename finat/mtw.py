import numpy

import FIAT

from gem import Literal, ListTensor, partial_indexed

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
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
