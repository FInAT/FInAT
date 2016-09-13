from __future__ import absolute_import, print_function, division

from .finiteelementbase import FiniteElementBase
import gem


class VectorFiniteElement(FiniteElementBase):

    def __init__(self, element, dimension):
        r"""A Finite element whose basis functions have the form:

        .. math::

            \boldsymbol\phi_{\beta i} = \mathbf{e}_{\beta}\phi_i

        Where :math:`\{\mathbf{e}_\beta,\, \beta=0\ldots\mathrm{dim}\}` is
        the basis for :math:`\mathbb{R}^{\mathrm{dim}}` and
        :math:`\{\phi_i\}` is the basis for the corresponding scalar
        finite element space.

        :param element: The scalar finite element.
        :param dimension: The geometric dimension of the vector element.

        :math:`\boldsymbol\phi_{i\beta}` is, of course, vector-valued. If
        we subscript the vector-value with :math:`\alpha` then we can write:

        .. math::
           \boldsymbol\phi_{\alpha(i\beta)} = \delta_{\alpha\beta}\phi_i

        This form enables the simplification of the loop nests which
        will eventually be created, so it is the form we employ here."""
        super(VectorFiniteElement, self).__init__()

        self._cell = element._cell
        self._degree = element._degree

        self._dimension = dimension

        self._base_element = element

    @property
    def index_shape(self):
        return self._base_element.index_shape + (self._dimension,)

    @property
    def value_shape(self):
        return self._base_element.value_shape + (self._dimension,)

    def basis_evaluation(self, q, entity=None, derivative=0):
        r"""Produce the recipe for basis function evaluation at a set of points :math:`q`:

        .. math::
            \boldsymbol\phi_{\alpha (i \beta) q} = \delta_{\alpha \beta}\phi_{i q}

            \nabla\boldsymbol\phi_{(\alpha \gamma) (i \beta) q} = \delta_{\alpha \beta}\nabla\phi_{\gamma i q}
        """

        scalarbasis = self._base_element.basis_evaluation(q, entity, derivative)

        indices = tuple(gem.Index() for i in scalarbasis.shape)

        # Work out which of the indices are for what.
        qi = len(q.index_shape) + len(self._base_element.index_shape)
        d = derivative

        # New basis function and value indices.
        i = gem.Index(extent=self._dimension)
        vi = gem.Index(extent=self._dimension)

        new_indices = indices[:qi] + (i,) + indices[qi: len(indices) - d] + (vi,) + indices[len(indices) - d:]

        return gem.ComponentTensor(gem.Product(gem.Delta(i, vi),
                                               gem.Indexed(scalarbasis, indices)),
                                   new_indices)

    def __hash__(self):
        """VectorFiniteElements are equal if they have the same base element
        and dimension."""

        return hash((self._dimension, self._base_element))

    def __eq__(self, other):
        """VectorFiniteElements are equal if they have the same base element
        and dimension."""

        return self._dimension == other._dimension and\
            self._base_element == other._base_element
