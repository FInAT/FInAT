import FIAT
import numpy
from gem import ListTensor, Literal

from finat.aw import _facet_transform
from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


class JohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree):
        if degree != 1:
            raise ValueError("Degree must be 1 for Johnson-Mercier element")
        if Citations is not None:
            Citations().register("Gopalakrishnan2024")
        super(JohnsonMercier, self).__init__(FIAT.JohnsonMercier(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        numbf = self.space_dimension()
        V = numpy.eye(numbf, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        Vsub = _facet_transform(self.cell, 1, coordinate_mapping)
        fdofs = Vsub.shape[0]
        print(Vsub.shape)
        V[:fdofs, :fdofs] = Vsub

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.
        return ListTensor(V.T)


class ReducedJohnsonMercier(PhysicallyMappedElement, FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree):
        if degree != 1:
            raise ValueError("Degree must be 1 for Johnson-Mercier element")
        if Citations is not None:
            Citations().register("Gopalakrishnan2024")
        super(JohnsonMercier, self).__init__(FIAT.JohnsonMercier(cell, degree, reduced=True))

    def basis_transformation(self, coordinate_mapping):
        raise NotImplementedError("TODO")
