import FIAT
import numpy
from gem import ListTensor, Literal

from finat.aw import _facet_transform
from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


class JohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree, variant=None):
        if degree != 1:
            raise ValueError("Degree must be 1 for Johnson-Mercier element")
        if Citations is not None:
            Citations().register("Gopalakrishnan2024")
        super(JohnsonMercier, self).__init__(FIAT.JohnsonMercier(cell, degree, variant=variant))

    def basis_transformation(self, coordinate_mapping):
        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        Vsub = _facet_transform(self.cell, 1, coordinate_mapping)
        if ndof < numbf:
            sd = self.cell.get_spatial_dimension()
            edofs = self._element.entity_dofs()
            indices = []
            for entity in edofs[sd-1]:
                indices.extend(edofs[sd-1][entity][:sd])
            Vsub = Vsub[indices, :]

        m, n = Vsub.shape
        V[:m, :n] = Vsub

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.
        return ListTensor(V.T)


class ReducedJohnsonMercier(JohnsonMercier):  # symmetric matrix valued
    def __init__(self, cell, degree):
        super(ReducedJohnsonMercier, self).__init__(cell, degree, variant="divergence")

    def entity_dofs(self):
        top = sell.cell.get_topology()
        sd = self.cell.get_spatial_dimension()
        edofs = {{entity: [] for entity in top[dim]} for dim in top}
        dim = sd - 1
        cur = 0
        for entity in sorted(top[dim]):
            edofs[dim][entity] = list(range(cur, cur + sd))
            cur = cur + sd
        return edofs

    @property
    def index_shape(self):
        top = sell.cell.get_topology()
        sd = self.cell.get_spatial_dimension()
        num_facets = len(top[sd-1])
        return (sd * num_facets,)

    def space_dimension(self):
        top = sell.cell.get_topology()
        sd = self.cell.get_spatial_dimension()
        num_facets = len(top[sd-1])
        return sd * num_facets
