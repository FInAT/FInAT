from finiteelementbase import FiniteElementBase
from derivatives import div, grad, curl
from ast import Recipe, IndexSum, Delta
import indices


class VectorFiniteElement(FiniteElementBase):

    def __init__(self, element, dimension):
        r"""A Finite element whose basis functions have the form:

        .. math::

            \boldsymbol\phi_{\beta,i} = \mathbf{e}_{\beta}\phi_i

        Where :math:`\{\mathbf{e}_\beta,\, \beta=0\ldots\mathrm{dim}\}` is
        the basis for :math:`\mathbb{R}^{\mathrm{dim}}` and
        :math:`\{\phi_i\}` is the basis for the corresponding scalar
        finite element space.

        :param element: The scalar finite element.
        :param dimension: The geometric dimension of the vector element.

        :math:`\boldsymbol\phi_{i,\beta}` is, of course, vector-valued. If
        we subscript the vector-value with :math:`\alpha` then we can write:

        .. math::
           \boldsymbol\phi_{\alpha,(i,\beta)} = \delta_{\alpha,\beta}\phi_i

        This form enables the simplification of the loop nests which
        will eventually be created, so it is the form we employ here."""
        super(VectorFiniteElement, self).__init__()

        self._cell = element._cell
        self._degree = element._degree

        self._dimension = dimension

        self._base_element = element

    def basis_evaluation(self, points, kernel_data, derivative=None):
        r"""Produce the recipe for basis function evaluation at a set of points
:math:`q`:

        .. math::
           \boldsymbol\phi_{\alpha,(i,\beta),q} = \delta_{\alpha,\beta}\phi_{i,q}

        """

        # Produce the base scalar recipe
        sr = self._base_element.basis_evaluation(points, kernel_data,
                                                 derivative)
        phi = sr.expression
        d, b, p = sr.indices

        # Additional dimension index along the vector dimension. Note
        # to self: is this the right order or does this index come
        # after any derivative index?
        alpha = (indices.DimensionIndex(self._dimension),)
        # Additional basis function index along the vector dimension.
        beta = (indices.BasisFunctionIndex(self._dimension),)

        return Recipe((alpha + d, b + beta, p), Delta(alpha + beta, phi))

    def field_evaluation(self, field_var, points,
                         kernel_data, derivative=None):
        r"""Produce the recipe for the evaluation of a field f at a set of
points :math:`q`:

        .. math::
           \boldsymbol{f}_{\alpha,q} = \sum_i f_{i,\alpha}\phi_{i,q}

        """
        # Produce the base scalar recipe
        sr = self._base_element.basis_evaluation(points, kernel_data,
                                                 derivative)
        phi = sr.expression
        d, b, p = sr.indices

        # Additional basis function index along the vector dimension.
        alpha = (indices.DimensionIndex(self._dimension),)

        expression = IndexSum(b, field_var[b + alpha] * phi)

        return Recipe((alpha + d, (), p), expression)

    def moment_evaluation(self, value, weights, points,
                          kernel_data, derivative=None):
        r"""Produce the recipe for the evaluation of the moment of
        :math:`u_{\alpha,q}` against a test function :math:`v_{\beta,q}`.

        .. math::
           \int u_{\alpha,q} : \phi_{\alpha,(i,\beta),q}\, \mathrm{d}x =
           \sum_q u_{\alpha}\phi_{i,q}w_q

        Appropriate code is also generated in the more general cases
        where derivatives are involved and where the value contains
        test functions.
        """

        # Produce the base scalar recipe
        sr = self._base_element.basis_evaluation(points, kernel_data,
                                                 derivative)
        phi = sr.expression
        d, b, p = sr.indices

        beta = (indices.BasisFunctionIndex(self._dimension),)

        (d_, b_, p_) = value.indices
        psi = value.replace_indices(zip(d_ + p_, beta + d + p)).expression

        w = weights.kernel_variable("w", kernel_data)

        expression = IndexSum(d + p, psi * phi * w[p])

        return Recipe(((), b + beta + b_, ()), expression)

    def pullback(self, phi, kernel_data, derivative=None):

        if derivative is None:
            return phi
        elif derivative == grad:
            return None  # IndexSum(alpha, Jinv[:, alpha] * grad(phi)[:,alpha])
        else:
            raise ValueError(
                "Lagrange elements do not have a %s operation") % derivative
