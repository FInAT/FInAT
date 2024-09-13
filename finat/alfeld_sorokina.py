import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import VectorFiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


class AlfeldSorokina(PhysicallyMappedElement, VectorFiatElement):
    def __init__(self, cell, degree=2):
        if degree != 2:
            raise ValueError("Alfeld-Sorokina only defined for degree == 2")
        if Citations is not None:
            Citations().register("AlfeldSorokina2016")
        super().__init__(FIAT.AlfeldSorokina(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        sd = self.cell.get_dimension()
        top = self.cell.get_topology()
        # Jacobians at cell center
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        ndof = self.space_dimension()
        V = numpy.eye(ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        voffset = 1 + sd
        for v in sorted(top[0]):
            s = voffset * v
            for i in range(sd):
                for j in range(sd):
                    V[s+1+i, s+1+j] = J[j, i]

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            for k in range(sd):
                V[:, voffset*v+1+k] /= h[v]
        return ListTensor(V.T)
