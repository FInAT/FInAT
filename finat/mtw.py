import numpy

import FIAT

from gem import Literal, ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations
from finat.piola_mapped import normal_tangential_edge_transform
from copy import deepcopy


class MardalTaiWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3):
        if Citations is not None:
            Citations().register("Mardal2002")
        super().__init__(FIAT.MardalTaiWinther(cell, degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        fdofs = sd + 1
        reduced_dofs[sd][0] = []
        for f in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][f] = reduced_dofs[sd-1][f][:fdofs]
        self._entity_dofs = reduced_dofs
        self._space_dimension = fdofs * len(reduced_dofs[sd-1])

    def basis_transformation(self, coordinate_mapping):
        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        entity_dofs = self.entity_dofs()
        for f in sorted(entity_dofs[sd-1]):
            cur = entity_dofs[sd-1][f][0]
            V[cur+1, cur:cur+sd] = normal_tangential_edge_transform(self.cell, J, detJ, f)

        return ListTensor(V.T)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self._space_dimension,)

    def space_dimension(self):
        return self._space_dimension
