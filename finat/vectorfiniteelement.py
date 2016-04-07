from finiteelementbase import FiniteElementBase


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

    def basis_evaluation(self, q, kernel_data, derivative=None, pullback=True):
        r"""Produce the recipe for basis function evaluation at a set of points
:math:`q`:

        .. math::
            \boldsymbol\phi_{\alpha (i \beta) q} = \delta_{\alpha \beta}\phi_{i q}

            \nabla\boldsymbol\phi_{(\alpha \gamma) (i \beta) q} = \delta_{\alpha \beta}\nabla\phi_{\gamma i q}

            \nabla\times\boldsymbol\phi_{(i \beta) q} = \epsilon_{2 \beta \gamma}\nabla\phi_{\gamma i q} \qquad\textrm{(2D)}

            \nabla\times\boldsymbol\phi_{\alpha (i \beta) q} = \epsilon_{\alpha \beta \gamma}\nabla\phi_{\gamma i q} \qquad\textrm{(3D)}

            \nabla\cdot\boldsymbol\phi_{(i \beta) q} = \nabla\phi_{\beta i q}
        """
        raise NotImplementedError

    def __hash__(self):
        """VectorFiniteElements are equal if they have the same base element
        and dimension."""

        return hash((self._dimension, self._base_element))

    def __eq__(self, other):
        """VectorFiniteElements are equal if they have the same base element
        and dimension."""

        return self._dimension == other._dimension and\
            self._base_element == other._base_element
