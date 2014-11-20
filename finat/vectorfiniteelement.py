from finiteelementbase import FiniteElementBase
from derivatives import div, grad, curl
from ast import Recipe, IndexSum, Delta, LeviCivita
import indices


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

        # Produce the base scalar recipe. The scalar basis can only
        # take a grad. For other derivatives, we need to do the
        # transform here.
        sr = self._base_element.basis_evaluation(q, kernel_data,
                                                 derivative and grad,
                                                 pullback)
        phi = sr.body
        d, b, p = sr.indices

        # Additional basis function index along the vector dimension.
        beta = (indices.BasisFunctionIndex(self._dimension),)

        if derivative is div:

            return Recipe((d[:-1], b + beta, p),
                          sr.replace_indices({d[-1]: beta[0]}).body)

        elif derivative is curl:
            if self.dimension == 2:

                return Recipe((d[:-1], b + beta, p), LeviCivita((2,) + beta, d[-1:], phi))
            elif self.dimension == 3:
                alpha = (indices.DimensionIndex(self._dimension),)

                return Recipe((d[:-1] + alpha, b + beta, p), LeviCivita(alpha + beta, d[-1:], phi))
            else:
                raise NotImplementedError

        else:
            # Additional dimension index along the vector dimension. Note
            # to self: is this the right order or does this index come
            # after any derivative index?
            alpha = (indices.DimensionIndex(self._dimension),)

            return Recipe((alpha + d, b + beta, p), Delta(alpha + beta, phi))

    def field_evaluation(self, field_var, q,
                         kernel_data, derivative=None, pullback=True):
        r"""Produce the recipe for the evaluation of a field f at a set of
points :math:`q`:

        .. math::
           \boldsymbol{f}_{\alpha q} = \sum_i f_{i \alpha}\phi_{i q}

           \nabla\boldsymbol{f}_{\alpha \beta q} = \sum_i f_{i \alpha}\nabla\phi_{\beta i q}

           \nabla\times\boldsymbol{f}_{q} = \epsilon_{2 \beta \gamma}\sum_{i} f_{i \beta}\nabla\phi_{\gamma i q} \qquad\textrm{(2D)}

           \nabla\times\boldsymbol{f}_{\alpha q} = \epsilon_{\alpha \beta \gamma}\sum_{i} f_{i \beta}\nabla\phi_{\gamma i q} \qquad\textrm{(3D)}

           \nabla\cdot\boldsymbol{f}_{q} = \sum_{i \alpha} f_{i \alpha}\nabla\phi_{\alpha i q}
        """
        # Produce the base scalar recipe. The scalar basis can only
        # take a grad. For other derivatives, we need to do the
        # transform here.
        sr = self._base_element.basis_evaluation(q, kernel_data,
                                                 derivative and grad,
                                                 pullback)
        phi = sr.body
        d, b, p = sr.indices

        if derivative is div:

            expression = IndexSum(b + d[-1:], field_var[b + d[-1:]] * phi)

            return Recipe((d[:-1], (), p), expression)

        elif derivative is curl:
            if self.dimension == 2:
                beta = (indices.BasisFunctionIndex(self._dimension),)

                expression = LeviCivita((2,), beta + d[-1:],
                                        IndexSum(b, field_var[b + beta] * phi))

                return Recipe((d[:-1], (), p), expression)
            elif self.dimension == 3:
                # Additional basis function index along the vector dimension.
                alpha = (indices.DimensionIndex(self._dimension),)
                beta = (indices.BasisFunctionIndex(self._dimension),)

                expression = LeviCivita(alpha, beta + d[-1:], IndexSum(b, field_var[b + beta] * phi))

                return Recipe((d[:-1] + alpha, (), p), expression)
            else:
                raise NotImplementedError
        else:
            # Additional basis function index along the vector dimension.
            alpha = (indices.DimensionIndex(self._dimension),)

            expression = IndexSum(b, field_var[b + alpha] * phi)

            return Recipe((alpha + d, (), p), expression)

    def moment_evaluation(self, value, weights, q,
                          kernel_data, derivative=None, pullback=True):
        r"""Produce the recipe for the evaluation of the moment of
        :math:`u_{\alpha,q}` against a test function :math:`v_{\beta,q}`.

        .. math::
           \int \boldsymbol{u} \cdot \boldsymbol\phi_{(i \beta)}\  \mathrm{d}x =
           \sum_q \boldsymbol{u}_{\beta q}\phi_{i q}w_q

           \int \boldsymbol{u}_{(\alpha \gamma)} \nabla \boldsymbol\phi_{(\alpha \gamma) (i \beta)}\  \mathrm{d}x =
           \sum_{\gamma q} \boldsymbol{u}_{(\beta \gamma) q}\nabla\phi_{\gamma i q}w_q

           \int u \nabla \times \boldsymbol\phi_{(i \beta)}\  \mathrm{d}x =
           \sum_{q} u_{q}\epsilon_{2\beta\gamma}\nabla\phi_{\gamma i q}w_q \qquad\textrm{(2D)}

           \int u_{\alpha} \nabla \times \boldsymbol\phi_{\alpha (i \beta)}\  \mathrm{d}x =
           \sum_{\alpha q} u_{\alpha q}\epsilon_{\alpha\beta\gamma}\nabla\phi_{\gamma i q}w_q \qquad\textrm{(3D)}

           \int u \nabla \cdot \boldsymbol\phi_{(i \beta)}\  \mathrm{d}x =
           \sum_{q} u_q\nabla\phi_{\beta i q}w_q

        Appropriate code is also generated where the value contains
        trial functions.
        """

        # Produce the base scalar recipe. The scalar basis can only
        # take a grad. For other derivatives, we need to do the
        # transform here.
        sr = self._base_element.basis_evaluation(q, kernel_data,
                                                 derivative and grad,
                                                 pullback)

        phi = sr.body
        d, b, p = sr.indices

        beta = (indices.BasisFunctionIndex(self._dimension),)

        (d_, b_, p_) = value.indices

        w = weights.kernel_variable("w", kernel_data)

        if derivative is div:
            beta = d[-1:]

            psi = value.replace_indices(zip(d_ + p_, d[:-1] + p)).body

            expression = IndexSum(d[:-1] + p, psi * phi * w[p])

        elif derivative is curl:
            if self.dimension == 2:

                beta = (indices.BasisFunctionIndex(self._dimension),)
                gamma = d[-1:]

                psi = value.replace_indices((d_ + p_, d[:-1] + p)).body

                expression = IndexSum(p, psi * LeviCivita((2,) + beta, gamma, phi) * w[p])
            elif self.dimension == 3:

                alpha = d_[-1:]
                beta = (indices.BasisFunctionIndex(self._dimension),)
                gamma = d[-1:]

                psi = value.replace_indices((d_[:-1] + p_, d[:-1] + p)).body

                expression = IndexSum(alpha + p, psi * LeviCivita(alpha + beta, gamma, phi) * w[p])
            else:
                raise NotImplementedError
        else:
            beta = (indices.BasisFunctionIndex(self._dimension),)

            psi = value.replace_indices(zip(d_ + p_, beta + d + p)).body

            expression = IndexSum(d + p, psi * phi * w[p])

        return Recipe(((), b + beta + b_, ()), expression)
