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
        self._indices = slice(None, None)
        super(JohnsonMercier, self).__init__(FIAT.JohnsonMercier(cell, degree, variant=variant))

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

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.
        return ListTensor(V.T)


class ReducedJohnsonMercier(JohnsonMercier):  # symmetric matrix valued
    def __init__(self, cell, degree, variant=None):
        super(ReducedJohnsonMercier, self).__init__(cell, degree, variant=variant)

        full_dofs = self._element.entity_dofs()
        top = cell.get_topology()
        sd = cell.get_spatial_dimension()
        fdim = sd - 1

        indices = []
        reduced_dofs = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}
        cur = 0
        for entity in sorted(top[fdim]):
            indices.extend(full_dofs[fdim][entity][:sd])
            reduced_dofs[fdim][entity] = list(range(cur, cur + sd))
            cur = cur + sd

        self._indices = indices
        self._entity_dofs = reduced_dofs
        self._space_dimension = cur

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension

    @property
    def index_shape(self):
        return (self._space_dimension,)
