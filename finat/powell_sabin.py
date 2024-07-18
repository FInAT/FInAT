import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


class QuadraticPowellSabin6(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2, avg=False):
        if degree != 2:
            raise ValueError("Degree must be 2 for PS6")
        self.avg = avg
        # if Citations is not None:
        #     Citations().register("PowellSabin")
        super().__init__(FIAT.QuadraticPowellSabin6(cell))

    def basis_transformation(self, coordinate_mapping):
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        h = coordinate_mapping.cell_size()

        d = self.cell.get_dimension()
        numbf = self.space_dimension()

        M = numpy.eye(numbf, dtype=object)

        for multiindex in numpy.ndindex(M.shape):
            M[multiindex] = Literal(M[multiindex])

        cur = 0
        for i in range(d+1):
            cur += 1  # skip the vertex
            J = Js[i]
            for j in range(d):
                for k in range(d):
                    M[cur+j, cur+k] = J[j, k] / h[i]
            cur += d

        return ListTensor(M)
