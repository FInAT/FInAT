"""Implementation of the Hu-Zhang finite elements."""
import numpy
import FIAT
from gem import Literal, ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations
from finat.aw import _facet_transform, _evaluation_transform


class HuZhang(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3, variant="integral"):
        if Citations is not None:
            Citations().register("Hu2015")
        super(HuZhang, self).__init__(FIAT.HuZhang(cell, degree, variant=variant))

    def basis_transformation(self, coordinate_mapping):
        p = self.degree
        sd = self.cell.get_spatial_dimension()
        ndof = self.space_dimension()
        numbf = self._element.space_dimension()
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        W = _evaluation_transform(self.cell, coordinate_mapping)

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W
        nverts = sd + 1
        ncomp = (sd*(sd+1)) // 2
        vdofs = nverts * ncomp

        ncomp = sd
        nfacets = sd + 1
        dofs_per_facet = p - 1
        fdofs = ncomp * nfacets * dofs_per_facet
        V[vdofs:vdofs+fdofs, vdofs:vdofs+fdofs] = _facet_transform(self.cell, p-2, coordinate_mapping)

        # internal DOFs
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])
        ########## Can be done right later. Putting this as a temporary thing, which presumably gives better conditioning than having diagonal 1s in this block.
        num_interior_dof_triples = (p*(p - 1))//2 # NOTE divided by 3 since the DOFs come in 3s below
        for j in range(num_interior_dof_triples):
            s = vdofs + fdofs + 3 * j
            #V[s:s+3, s:s+3] = W / detJ

#        # RESCALING FOR CONDITIONING
#        h = coordinate_mapping.cell_size()
#
#        for e in range(3):
#            eff_h = h[e]
#            for i in range(24):
#                V[i, 3*e] = V[i, 3*e]/(eff_h*eff_h)
#                V[i, 1 + 3*e] = V[i, 1 + 3*e]/(eff_h*eff_h)
#                V[i, 2 + 3*e] = V[i, 2 + 3*e]/(eff_h*eff_h)

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.

        return ListTensor(V.T)
