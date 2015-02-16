import pymbolic.primitives as p
import inspect
import FIAT
from derivatives import grad
from ast import Let, Array, Inverse, Det


class Kernel(object):
    def __init__(self, recipe, kernel_data):
        """An object bringing together a :class:`~.ast.Recipe` and its
        corresponding :class:`~.utils.KernelData` context.
        """

        self.recipe = recipe
        self.kernel_data = kernel_data


class KernelData(object):
    def __init__(self, coordinate_element, coordinate_var=None, affine=None):
        """
        :param coordinate_element: the (vector-valued) finite element for
            the coordinate field.
        :param coordinate_var: the symbolic variable for the
            coordinate values. If no pullbacks are present in the kernel,
            this may be omitted.
        :param affine: Specifies whether the pullback is affine (and therefore
            whether the Jacobian must be evaluated at every quadrature point).
            If not specified, this is inferred from coordinate_element.
        """

        self.coordinate_element = coordinate_element
        self.coordinate_var = coordinate_var
        if affine is None:
            self.affine = coordinate_element.degree <= 1 and \
                isinstance(coordinate_element.cell, _simplex)
        else:
            self.affine = affine

        self.static = {}
        # The set of undefined symbols in this kernel.
        self.kernel_args = set(coordinate_var) if coordinate_var else set()
        self.geometry = {}
        self.variables = set()

        #: The geometric dimension of the physical space.
        self.gdim = coordinate_element._dimension
        #: The topological dimension of the reference element
        self.tdim = coordinate_element._cell.get_spatial_dimension()

        self._variable_count = 0
        self._point_count = 0
        self._wt_count = 0
        self._variable_cache = {}

    def tabulation_variable_name(self, element, points):
        """Given a finite element and a point set, return a variable name phi_n
        where n is guaranteed to be unique to that combination of element and
        points."""

        key = (id(element), id(points))

        try:
            return self._variable_cache[key]
        except KeyError:
            name = u'\u03C6_'.encode("utf-8") \
                   + str(self._variable_count)
            self._variable_cache[key] = name
            self._variable_count += 1
            self.variables.add(name)
            return self._variable_cache[key]

    def point_variable_name(self, points):
        """Given a point set, return a variable name xi_n
        where n is guaranteed to be unique to that set of
        points."""

        key = (id(points),)

        try:
            return self._variable_cache[key]
        except KeyError:
            name = u'\u03BE_'.encode("utf-8") \
                   + str(self._point_count)
            self._variable_cache[key] = name
            self._point_count += 1
            self.variables.add(name)
            return self._variable_cache[key]

    def weight_variable_name(self, weights):
        """Given an iterable of weights set, return a variable name wt_n
        where n is guaranteed to be unique to that set of weights."""

        key = (id(weights),)

        try:
            return self._variable_cache[key]
        except KeyError:
            name = u'\u03BE_'.encode("utf-8") \
                   + str(self._wt_count)
            self._variable_cache[key] = name
            self._wt_count += 1
            self.variables.add(name)
            return self._variable_cache[key]

    def new_variable(self, prefix=None):
        """Create a variable guaranteed to be unique in the kernel context."""
        name = prefix or "tmp"
        if name not in self.variables:
            self.variables.add(name)
            return p.Variable(name)

        # Prefix was already in use, so append an index.
        i = 0
        while True:
            varname = "%s_%d" % (name, i)
            if varname not in self.variables:
                self.variables.add(varname)
                return p.Variable(varname)
            i += 1

    @property
    def J(self):
        '''The Jacobian of the coordinate transformation.

        .. math::

            J_{\gamma,\tau} = \frac{\partial x_\gamma}{\partial X_\tau}

        Where :math:`x` is the physical coordinate and :math:`X` is the
        local coordinate.
        '''
        try:
            return self.geometry["J"]
        except KeyError:
            self.geometry["J"] = Array("J", (self.gdim, self.tdim))
            return self.geometry["J"]

    @property
    def invJ(self):
        '''The Moore-Penrose pseudo-inverse of the coordinate transformation.

        .. math::

            J^{-1}_{\tau,\gamma} = \frac{\partial X_\tau}{\partial x_\gamma}

        Where :math:`x` is the physical coordinate and :math:`X` is the
        local coordinate.
        '''

        try:
            return self.geometry["invJ"]
        except KeyError:
            self.geometry["invJ"] = Array("invJ", (self.tdim, self.gdim))
            return self.geometry["invJ"]

    @property
    def detJ(self):
        '''The determinant of the coordinate transformation.'''

        try:
            return self.geometry["detJ"]
        except KeyError:
            self.geometry["detJ"] = p.Variable("detJ")
            return self.geometry["detJ"]

    def bind_geometry(self, expression, points=None):
        """Let statement defining the geometry for expression. If no geometry
        is required, return expression."""

        if len(self.geometry) == 0:
            return expression

        g = self.geometry

        if points is None:
            points = self._origin

        inner_lets = []
        if "invJ" in g:
            inner_lets += (g["invJ"], Inverse(g["J"]))
        if "detJ" in g:
            inner_lets += (g["detJ"], Det(g["J"]))

        J_expr = self.coordinate_element.evaluate_field(
            self.coordinate_var, points, self, derivative=grad, pullback=False)
        if points:
            J_expr = J_expr.replace_indices(
                zip(J_expr.indices[-1], expression.indices[-1]))

        return Let((g["J"], J_expr),
                   Let(inner_lets, expression) if inner_lets else expression)


# Tuple of simplex cells. This relies on the fact that FIAT only
# defines simplex elements.
_simplex = tuple(e for e in FIAT.reference_element.__dict__.values()
                 if (inspect.isclass(e) and
                     issubclass(e, FIAT.reference_element.ReferenceElement) and
                     e is not FIAT.reference_element.ReferenceElement))
