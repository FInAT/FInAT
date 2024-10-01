import FIAT

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from finat.bernardi_raugel import bernardi_raugel_transformation
from copy import deepcopy


class ArnoldQin(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if degree != 2:
            raise ValueError("Arnold-Qin only defined for degree = 2")
        if Citations is not None:
            Citations().register("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        return bernardi_raugel_transformation(self, coordinate_mapping)


class ReducedArnoldQin(ArnoldQin):
    def __init__(self, cell, degree=2):
        super().__init__(cell, degree=degree)

        reduced_dofs = deepcopy(self._element.entity_dofs())
        sd = cell.get_spatial_dimension()
        cur = reduced_dofs[sd-1][0][0]
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = [cur]
            cur += 1
        self._entity_dofs = reduced_dofs

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    def space_dimension(self):
        return (self.cell.get_spatial_dimension() + 1)**2
