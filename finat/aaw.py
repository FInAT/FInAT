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

        for i in range(15):
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
