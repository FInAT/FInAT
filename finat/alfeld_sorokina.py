import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement
from finat.piola_mapped import adjugate


class AlfeldSorokina(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if degree != 2:
            raise ValueError("Alfeld-Sorokina only defined for degree = 2")
        if Citations is not None:
            Citations().register("AlfeldSorokina2016")
        super().__init__(FIAT.AlfeldSorokina(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)
        adjJ = adjugate([[J[i, j] for j in range(sd)] for i in range(sd)])

        ndof = self.space_dimension()
        V = numpy.eye(ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        edofs = self.entity_dofs()
        for dim in edofs:
            for entity in sorted(edofs[dim]):
                dofs = edofs[dim][entity]
                if dim == 0:
                    s = dofs[0]
                    V[s, s] = detJ
                    dofs = dofs[1:]
                for i in range(0, len(dofs), sd):
                    s = dofs[i:i+sd]
                    V[numpy.ix_(s, s)] = adjJ

        return ListTensor(V.T)
