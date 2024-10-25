"""Implementation of the Hu-Zhang finite elements."""
import numpy
import FIAT
from gem import Literal, ListTensor
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations
from finat.aw import _facet_transform, _evaluation_transform


class HuZhang(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3):
        if Citations is not None:
            Citations().register("Hu2015")
        super().__init__(FIAT.HuZhang(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        ndofs = self.space_dimension()
        V = numpy.eye(ndofs, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        W = _evaluation_transform(self.cell, coordinate_mapping)

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W
        voffset = 9

        Vsub = _facet_transform(self.cell, self.degree-2, coordinate_mapping)
        foffset = voffset + Vsub.shape[0]
        V[voffset:foffset, voffset:foffset] = Vsub

        # internal DOFs
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])
        offset = foffset
        while offset < ndofs:
            V[offset:offset+3, offset:offset+3] = W / detJ
            offset += 3
        return ListTensor(V.T)
