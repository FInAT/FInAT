from finiteelementbase import FiniteElementBase
from derivatives import div, grad, curl
from utils import doc_inherit
from ast import Recipe, IndexSum, Delta
import indices


class VectorFiniteElement(FiniteElementBase):

    def __init__(self, element, dimension):
        r"""A Finite element whose basis functions have the form:

        .. math::

            \boldsymbol\phi_{\alpha,i} = \mathbf{e}_{\alpha}\phi_i

        Where :math:`\{\mathbf{e}_\alpha,\, \alpha=0\ldots\mathrm{dim}\}` is
        the basis for :math:`\mathbb{R}^{\mathrm{dim}}` and
        :math:`\{\phi_i\}` is the basis for the corresponding scalar
        finite element space.

        :param element: The scalar finite element.
        :param dimension: The geometric dimension of the vector element.

        :math:`\boldsymbol\phi_{\alpha,i}` is, of course, vector-valued. If
        we subscript the vector-value with :math:`\beta` then we can write:

        .. math::
           \boldsymbol\phi_{\beta,(\alpha,i)} = \delta_{\beta,\alpha}\phi_i

        This form enables the simplification of the loop nests which
        will eventually be created, so it is the form we employ here.  """
        super(VectorFiniteElement, self).__init__()

        self._cell = element._cell
        self._degree = element._degree

        self._dimension = dimension

        self._base_element = element

    @doc_inherit
    def basis_evaluation(self, points, kernel_data, derivative=None):
        # This is incorrect. We only get the scalar value. We need to
        # bring in some sort of delta in order to get the right rank.

        # Produce the base scalar recipe
        sr = self._base_element.basis_evaluation(points, kernel_data, derivative)

        # Additional basis function index along the vector dimension.
        alpha = indices.BasisFunctionIndex(points.points.shape[1])
        # Additional dimension index along the vector dimension. Note
        # to self: is this the right order or does this index come
        # after any derivative index?
        beta = indices.DimensionIndex(points.points.shape[1])

        d, b, p = sr.split_indices

        return Recipe((beta,) + d + (alpha,) + b + p,
                      Delta((beta, alpha), sr),
                      sr.depends)

    @doc_inherit
    def field_evaluation(self, field_var, points,
                         kernel_data, derivative=None):

        basis = self._base_element.basis_evaluation(points,
                                                    kernel_data, derivative)

        alpha = indices.DimensionIndex(points.points.shape[1])

        d, b, p = basis.split_indices

        free_ind = (alpha,) + d + p

        i = b[0]

        instructions = IndexSum(i, field_var[i, alpha] * basis)

        depends = [field_var]

        return Recipe(free_ind, instructions, depends)

    @doc_inherit
    def moment_evaluation(self, value, weights, points,
                          kernel_data, derivative=None):

        basis = self._base_element.basis_evaluation(self, points,
                                                    kernel_data, derivative)
        w = weights.kernel_variable("w", kernel_data)
        ind = basis.indices

        q = ind[-1]
        alpha = indices.DimensionIndex(points.points.shape[1])

        if derivative is None:
            sum_ind = [q]
        elif derivative == grad:
            sum_ind = [ind[0], q]
        else:
            raise NotImplementedError()

        value_ind = [alpha] + sum_ind

        instructions = [IndexSum(sum_ind, value[value_ind] * w[q] * basis[ind])]

        depends = [value]

        free_ind = [alpha, ind[-2]]

        return Recipe(free_ind, instructions, depends)

    @doc_inherit
    def pullback(self, phi, kernel_data, derivative=None):

        if derivative is None:
            return phi
        elif derivative == grad:
            return None  # IndexSum(alpha, Jinv[:, alpha] * grad(phi)[:,alpha])
        else:
            raise ValueError(
                "Lagrange elements do not have a %s operation") % derivative
