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

        #: The geometric dimension of the physical space.
        self.gdim = coordinate_element._dimension

        #: The topological dimension of the reference element
        self.tdim = coordinate_element._cell.get_spatial_dimension()

        self._variable_count = 0
        self._variable_cache = {}

    def tabulation_variable_name(self, element, points):
        """Given a finite element and a point set, return a variable name phi_n
        where n is guaranteed to be unique to that combination of element and
        points."""

        key = (id(element), id(points))

        try:
            return self._variable_cache[key]
        except KeyError:
            self._variable_cache[key] = u'\u03C6_'.encode("utf-8") \
                                        + str(self._variable_count)
            self._variable_count += 1
            return self._variable_cache[key]

    def J(self, points):

        try:
            return self.geometry["J"]
        except KeyError:
            self.geometry["J"] = p.Variable("J")
            return self.geometry["J"]

    def invJ(self, points):

        # ensure J exists
        self.J(points)

        try:
            return self.geometry["invJ"]
        except KeyError:
            self.geometry["invJ"] = p.Variable("invJ")
            return self.geometry["invJ"]

    def detJ(self, points):

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
