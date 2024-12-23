import FIAT
from gem import ListTensor

from finat.argyris import _edge_transform
from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement


class QuadraticPowellSabin6(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("PowellSabin1977")
        super().__init__(FIAT.QuadraticPowellSabin6(cell))

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


class QuadraticPowellSabin12(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree=2, avg=False):
        self.avg = avg
        if Citations is not None:
            Citations().register("PowellSabin1977")
        super().__init__(FIAT.QuadraticPowellSabin12(cell))

    def basis_transformation(self, coordinate_mapping):
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        V = identity(self.space_dimension())

        sd = self.cell.get_dimension()
        top = self.cell.get_topology()
        voffset = sd + 1
        for v in sorted(top[0]):
            s = voffset * v
            for i in range(sd):
                for j in range(sd):
                    V[s+1+i, s+1+j] = J[j, i]

        _edge_transform(V, 1, 0, self.cell, coordinate_mapping, avg=self.avg)

        # Patch up conditioning
        h = coordinate_mapping.cell_size()
        for v in sorted(top[0]):
            for k in range(sd):
                V[:, voffset*v+1+k] /= h[v]
        return ListTensor(V.T)
