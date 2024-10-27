import FIAT
import numpy
from gem import ListTensor, Literal

from finat.aw import _facet_transform, _evaluation_transform
from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


class JohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree=1, variant=None):
        if Citations is not None:
            Citations().register("Gopalakrishnan2024")
        self._indices = slice(None, None)
        super().__init__(FIAT.JohnsonMercier(cell, degree, variant=variant))

    def basis_transformation(self, coordinate_mapping):
        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        Vsub = _facet_transform(self.cell, 1, coordinate_mapping)
        Vsub = Vsub[:, self._indices]
        m, n = Vsub.shape
        V[:m, :n] = Vsub
        cur = m

        # internal DOFs
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        detJ = coordinate_mapping.detJ_at(bary)
        W = _evaluation_transform(self.cell, coordinate_mapping)
        ncomp = W.shape[0]
        while cur < ndof:
            V[cur:cur+ncomp, cur:cur+ncomp] = W / detJ
            cur += ncomp

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.
        return ListTensor(V.T)
