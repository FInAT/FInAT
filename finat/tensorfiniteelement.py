from __future__ import absolute_import, print_function, division

from .finiteelementbase import FiniteElementBase
import gem


class TensorFiniteElement(FiniteElementBase):

    def __init__(self, element, shape):
        # TODO: Update docstring for arbitrary rank!
        r"""A Finite element whose basis functions have the form:

        .. math::

            \boldsymbol\phi_{i \alpha \beta} = \mathbf{e}_{\alpha} \mathbf{e}_{\beta}^{\mathrm{T}}\phi_i

        Where :math:`\{\mathbf{e}_\alpha,\, \alpha=0\ldots\mathrm{shape[0]}\}` and
        :math:`\{\mathbf{e}_\beta,\, \beta=0\ldots\mathrm{shape[1]}\}` are
        the bases for :math:`\mathbb{R}^{\mathrm{shape[0]}}` and
        :math:`\mathbb{R}^{\mathrm{shape[1]}}` respectively; and
        :math:`\{\phi_i\}` is the basis for the corresponding scalar
        finite element space.

        :param element: The scalar finite element.
        :param shape: The geometric shape of the tensor element.

        :math:`\boldsymbol\phi_{i\alpha\beta}` is, of course, tensor-valued. If
        we subscript the vector-value with :math:`\gamma\epsilon` then we can write:

        .. math::
           \boldsymbol\phi_{\gamma\epsilon(i\alpha\beta)} = \delta_{\gamma\alpha}\delta{\epsilon\beta}\phi_i

        This form enables the simplification of the loop nests which
        will eventually be created, so it is the form we employ here."""
        super(TensorFiniteElement, self).__init__()

        self._cell = element._cell
        self._degree = element._degree

        self._shape = shape

        self._base_element = element

    @property
    def base_element(self):
        """The base element of this tensor element."""
        return self._base_element

    @property
    def index_shape(self):
        return self._base_element.index_shape + self._shape

    @property
    def value_shape(self):
        return self._base_element.value_shape + self._shape

    def basis_evaluation(self, q, entity=None, derivative=0):
        r"""Produce the recipe for basis function evaluation at a set of points :math:`q`:

        .. math::
            \boldsymbol\phi_{(\gamma \epsilon) (i \alpha \beta) q} = \delta_{\alpha \gamma}\delta{\beta \epsilon}\phi_{i q}

            \nabla\boldsymbol\phi_{(\epsilon \gamma \zeta) (i \alpha \beta) q} = \delta_{\alpha \epsilon} \deta{\beta \gamma}\nabla\phi_{\zeta i q}
        """

        scalarbasis = self._base_element.basis_evaluation(q, entity, derivative)

        indices = tuple(gem.Index() for i in scalarbasis.shape)

        # Work out which of the indices are for what.
        qi = len(q.index_shape) + len(self._base_element.index_shape)
        d = derivative

        # New basis function and value indices.
        i = tuple(gem.Index(extent=d) for d in self._shape)
        vi = tuple(gem.Index(extent=d) for d in self._shape)

        new_indices = indices[:qi] + i + indices[qi: len(indices) - d] + vi + indices[len(indices) - d:]

        return gem.ComponentTensor(gem.Product(reduce(gem.Product,
                                                      (gem.Delta(j, k)
                                                       for j, k in zip(i, vi))),
                                               gem.Indexed(scalarbasis, indices)),
                                   new_indices)

    def __hash__(self):
        """TensorFiniteElements are equal if they have the same base element
        and shape."""
        return hash((self._shape, self._base_element))

    def __eq__(self, other):
        """TensorFiniteElements are equal if they have the same base element
        and shape."""
        return type(self) == type(other) and self._shape == other._shape and \
            self._base_element == other._base_element
