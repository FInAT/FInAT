from .points import PointSet
from .indices import PointIndex, PointIndexBase
from .ast import Let, Det, Inverse, Recipe
from .mappers import IdentityMapper
from .derivatives import grad


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
            q = PointIndex(PointSet([[0] * self.kernel_data.tdim]))

            body = self._bind_geometry(q, body)

        elif self.local_geometry:
            for s in self.local_geometry:
                s.set_error()
            print expr
            raise ValueError("Unbound local geometry in tree")

        # Reconstruct the recipe
        return expr.__class__(self.rec(expr.indices),
                              body)

    def map_variable(self, var):

        if var.name in ("J", "invJ", "detJ"):
            self.local_geometry.add(var)

        return var

    def map_index_sum(self, expr):

        body = self.rec(expr.body)

        if not self.kernel_data.affine \
           and self.local_geometry \
           and isinstance(expr.indices[-1], PointIndexBase):
            q = expr.indices[-1]

            body = self._bind_geometry(q, body)

        # Reconstruct the index_sum
        return expr.__class__(self.rec(expr.indices),
                              body)

    def _bind_geometry(self, q, body):

        kd = self.kernel_data

        # Note that this may not be safe for tensor product elements.
        phi_x = kd.coordinate_var
        element = kd.coordinate_element
        J = element.field_evaluation(phi_x, q, kd, grad, pullback=False)

        d, b, q = J.indices
        # In the affine case, there is only one point. In the
        # non-affine case, binding the point index is the problem of
        # kernel as a whole
        if self.kernel_data.affine:
            J = J.replace_indices(zip(q, (0,)))
        J.indices = (d, b, ())

        inner_lets = ((kd.detJ, Det(kd.J)),) if kd.detJ in self.local_geometry else ()
        inner_lets += ((kd.invJ, Inverse(kd.J)),) if kd.invJ in self.local_geometry else ()

        # The local geometry goes out of scope at this point.
        self.local_geometry = set()

        return Let(((kd.J, J),),
                   Let(inner_lets, body) if inner_lets else body)
