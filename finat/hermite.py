import FIAT
from gem import ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=3):
        if Citations is not None:
            Citations().register("Ciarlet1972")
        super().__init__(FIAT.CubicHermite(cell))

    def basis_transformation(self, coordinate_mapping):
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        h = coordinate_mapping.cell_size()

        d = self.cell.get_dimension()
        M = identity(self.space_dimension())

        cur = 0
        for i in range(d+1):
            cur += 1  # skip the vertex
            J = Js[i]
            for j in range(d):
                for k in range(d):
                    M[cur+j, cur+k] = J[j, k] / h[i]
            cur += d

        return ListTensor(M)
