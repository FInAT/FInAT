import FIAT
import numpy
from gem import ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement
from finat.piola_mapped import piola_inverse


class AlfeldSorokina(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("AlfeldSorokina2016")
        super().__init__(FIAT.AlfeldSorokina(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        dofs = self.entity_dofs()
        V = identity(self.space_dimension())

        # Undo the Piola transform
        nodes = self._element.get_dual_set().nodes
        Finv = piola_inverse(self.cell, J, detJ)
        for dim in sorted(dofs):
            for e in sorted(dofs[dim]):
                k = 0
                while k < len(dofs[dim][e]):
                    cur = dofs[dim][e][k]
                    if len(nodes[cur].deriv_dict) > 0:
                        V[cur, cur] = detJ
                        k += 1
                    else:
                        s = dofs[dim][e][k:k+sd]
                        V[numpy.ix_(s, s)] = Finv
                        k += sd

        return ListTensor(V.T)
