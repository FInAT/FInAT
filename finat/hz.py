"""Implementation of the Hu-Zhang finite elements."""
import FIAT
from gem import ListTensor
from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement
from finat.aw import _facet_transform, _evaluation_transform


class HuZhang(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3, variant=None):
        if Citations is not None:
            Citations().register("Hu2015")
        self.variant = variant
        super().__init__(FIAT.HuZhang(cell, degree, variant=variant))

    def basis_transformation(self, coordinate_mapping):
        ndofs = self.space_dimension()
        V = identity(ndofs)

        sd = self.cell.get_spatial_dimension()
        W = _evaluation_transform(self.cell, coordinate_mapping)

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W
        ncomp = W.shape[0]
        num_verts = sd+1
        cur = num_verts * ncomp

        Vsub = _facet_transform(self.cell, self.degree-2, coordinate_mapping)
        fdofs = Vsub.shape[0]
        V[cur:cur+fdofs, cur:cur+fdofs] = Vsub
        cur += fdofs

        # internal DOFs
        if self.variant == "point":
            while cur < ndofs:
                V[cur:cur+ncomp, cur:cur+ncomp] = W
                cur += ncomp

        # RESCALING FOR CONDITIONING
        h = coordinate_mapping.cell_size()
        for e in range(num_verts):
            V[:, ncomp*e:ncomp*(e+1)] *= 1/(h[e] * h[e])

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.
        return ListTensor(V.T)
