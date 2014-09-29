from pymbolic.mapper import IdentityMapper
from indices import PointIndex
from points import PointSet
from ast import Let, Det, Inverse, Recipe
from derivatives import grad


class GeometryMapper(IdentityMapper):
    """A mapper which identifies Jacobians, inverse Jacobians and their
    determinants in expressions, and inserts the code to define them at
    the correct point in the tree."""

    def __init__(self, kernel_data):
        """
        :arg context: a mapping from variable names to values
        """
        super(GeometryMapper, self).__init__()

        self.local_geometry = set()

        self.kernel_data = kernel_data

    def __call__(self, expr, *args, **kwargs):
        """
        In the affine case we need to do geometry insertion at the top
        level.
        """

        if not isinstance(expr, Recipe):
            raise TypeError("Can only map geometry on a Recipe")

        body = self.rec(expr.body)

        if self.kernel_data.affine and self.local_geometry:
            # This is the bottom corner. We actually want the
            # circumcentre.
            q = PointIndex([[0] * self.kernel_data.tdim])

            body = self._bind_geometry(q, body)

        elif self.local_geometry:
            raise ValueError("Unbound local geometry in tree")

        # Reconstruct the recipe
        return expr.__class__(self.rec(expr.indices),
                              body)

    def map_variable(self, var):

        if var.name in ("J", "invJ", "detJ"):
            self.local_geometry.add(var)

        return self

    def map_index_sum(self, expr):

        body = self.rec(expr.body)

        if not self.kernel_data.affine \
           and self.local_geometry \
           and isinstance(self.indices[-1], PointIndex):
            q = self.indices[-1]

            body = self._bind_geometry(q, body)

        # Reconstruct the index_sum
        return expr.__class__(self.rec(expr.indices),
                              body)

    def _bind_geometry(self, q, body):

        kd = self.kernel_data

        # Note that this may not be safe for tensor product elements.
        phi_x = kd.coordinate_var
        element = kd.coordinate_element
        J = element.field_evaluation(phi_x, q, kd, grad)

        inner_lets = (((kd.detJ, Det(kd.J)),)
                      if kd.detJ in self.local_geometry else ()
                      + ((kd.invJ, Inverse(kd.J)),)
                      if kd.invJ in self.local_geometry else ())

        # The local geometry goes out of scope at this point.
        self.local_geometry = set()

        return Let(((kd.J, J)),
                   Let(inner_lets, body) if inner_lets else body)
