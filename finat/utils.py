import pymbolic.primitives as p
import inspect
import FIAT


class KernelData(object):
    def __init__(self, coordinate_element, affine=None):
        """
        :param coordinate_element: the (vector-valued) finite element for
            the coordinate field.
        :param affine: Specifies whether the pullback is affine (and therefore
            whether the Jacobian must be evaluated at every quadrature point).
            If not specified, this is inferred from coordinate_element.
        """

        self.coordinate_element = coordinate_element
        if affine is None:
            self.affine = coordinate_element.degree <= 1 \
                and isinstance(coordinate_element.cell, _simplex)
        else:
            self.affine = True

        self.static = {}
        self.params = {}
        self.geometry = {}

    @property
    def J(self):

        try:
            return self.geometry["J"]
        except KeyError:
            self.geometry["J"] = p.Variable("J")
            return self.geometry["J"]

    @property
    def invJ(self):

        # ensure J exists
        self.J

        try:
            return self.geometry["invJ"]
        except KeyError:
            self.geometry["invJ"] = p.Variable("invJ")
            return self.geometry["invJ"]

    @property
    def detJ(self):

        # ensure J exists
        self.J

        try:
            return self.geometry["detJ"]
        except KeyError:
            self.geometry["detJ"] = p.Variable("detJ")
            return self.geometry["detJ"]

# Tuple of simplex cells. This relies on the fact that FIAT only
# defines simplex elements.
_simplex = tuple(e for e in FIAT.reference_element.__dict__.values()
                 if (inspect.isclass(e)
                     and issubclass(e, FIAT.reference_element.ReferenceElement)
                     and e is not FIAT.reference_element.ReferenceElement))
