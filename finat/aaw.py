import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class ArnoldAwanouWinther(FiatElement):
    def __init__(self, cell, degree):
        super(ArnoldAwanouWinther, self).__init__(FIAT.ArnoldAwanouWinther(cell, degree))



"""
class ArnoldAwanouWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        if degree != 2:
            raise ValueError("Degree must be 2 for Arnold-Awanou-Winther element")

        super().__init__(FIAT.ArnoldAwanouWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        V = numpy.zeros((18, 15), dtype=object)

        for i in range(18):
            for j in range(15):
                V[i, j] = Literal(0)

        for i in range(15):
            V[i, i] = Literal(1)

        return ListTensor(V.T)

    # This wipes out the constraint dofs. FIAT gives a 18 DOF element
    # because we need some extra functions to enforce that sigma_nn
    # is linear on each edge.
    # However, we only have a 15 DOF element.
    def entity_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]},
                2: {0: [12, 13, 14]}}

    @property
    def index_shape(self):
        import ipdb; ipdb.set_trace()
        return (15,)

    def space_dimension(self):
        return 15
"""
