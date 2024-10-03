import FIAT

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from finat.bernardi_raugel import bernardi_raugel_transformation
from copy import deepcopy


class GuzmanNeilan(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=None):
        sd = cell.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError("Guzman-Neilan only defined for degree = dim")
        if Citations is not None:
            Citations().register("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilan(cell, degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        cur = reduced_dofs[sd-1][0][0]
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = [cur]
            cur += 1
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        return bernardi_raugel_transformation(self, coordinate_mapping)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    def space_dimension(self):
        return (self.cell.get_spatial_dimension() + 1)**2
