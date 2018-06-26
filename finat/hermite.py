import numpy

import FIAT
from gem import Indexed, Literal, ListTensor

from finat.fiat_elements import ScalarFiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations


class Hermite(PhysicallyMappedElement, ScalarFiatElement):
    def __init__(self, cell, degree):
        if degree != 3:
            raise ValueError("Degree must be 3 for Hermite element")
        if Citations is not None:
            Citations().register("Ciarlet1972")
        super().__init__(FIAT.CubicHermite(cell))

    def basis_transformation(self, coordinate_mapping):
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        d = self.cell.get_dimension()
        numbf = self.space_dimension()

        def n(J):
            assert J.shape == (d, d)
            return numpy.array(
                [[Indexed(J, (i, j)) for j in range(d)]
                 for i in range(d)])

        M = numpy.eye(numbf, dtype=object)

        for multiindex in numpy.ndindex(M.shape):
            M[multiindex] = Literal(M[multiindex])

        cur = 0
        for i in range(d+1):
            cur += 1  # skip the vertex
            M[cur:cur+d, cur:cur+d] = n(Js[i])
            cur += d

        return ListTensor(M)
