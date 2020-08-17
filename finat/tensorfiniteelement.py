from functools import reduce

import numpy

import gem

from finat.finiteelementbase import FiniteElementBase


class TensorFiniteElement(FiniteElementBase):

    def __init__(self, element, shape, transpose=False):
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
        :param transpose: Changes the DoF ordering from the
                          Firedrake-style XYZ XYZ XYZ XYZ to the
                          FEniCS-style XXXX YYYY ZZZZ.  That is,
                          tensor shape indices come before the scalar
                          basis function indices when transpose=True.

        :math:`\boldsymbol\phi_{i\alpha\beta}` is, of course, tensor-valued. If
        we subscript the vector-value with :math:`\gamma\epsilon` then we can write:

        .. math::
           \boldsymbol\phi_{\gamma\epsilon(i\alpha\beta)} = \delta_{\gamma\alpha}\delta_{\epsilon\beta}\phi_i

        This form enables the simplification of the loop nests which
        will eventually be created, so it is the form we employ here."""
        super(TensorFiniteElement, self).__init__()
        self._base_element = element
        self._shape = shape
        self._transpose = transpose

    @property
    def base_element(self):
        """The base element of this tensor element."""
        return self._base_element

    @property
    def cell(self):
        return self._base_element.cell

    @property
    def degree(self):
        return self._base_element.degree

    @property
    def formdegree(self):
        return self._base_element.formdegree

    def entity_dofs(self):
        raise NotImplementedError("No one uses this!")

    def space_dimension(self):
        return int(numpy.prod(self.index_shape))

    @property
    def index_shape(self):
        if self._transpose:
            return self._shape + self._base_element.index_shape
        else:
            return self._base_element.index_shape + self._shape

    @property
    def value_shape(self):
        return self._shape + self._base_element.value_shape

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        r"""Produce the recipe for basis function evaluation at a set of points :math:`q`:

        .. math::
            \boldsymbol\phi_{(\gamma \epsilon) (i \alpha \beta) q} = \delta_{\alpha \gamma} \delta_{\beta \epsilon}\phi_{i q}

            \nabla\boldsymbol\phi_{(\epsilon \gamma \zeta) (i \alpha \beta) q} = \delta_{\alpha \epsilon} \delta_{\beta \gamma}\nabla\phi_{\zeta i q}
        """
        scalar_evaluation = self._base_element.basis_evaluation
        return self._tensorise(scalar_evaluation(order, ps, entity, coordinate_mapping=coordinate_mapping))

    def point_evaluation(self, order, point, entity=None):
        scalar_evaluation = self._base_element.point_evaluation
        return self._tensorise(scalar_evaluation(order, point, entity))

    def _tensorise(self, scalar_evaluation):
        # Old basis function and value indices
        scalar_i = self._base_element.get_indices()
        scalar_vi = self._base_element.get_value_indices()

        # New basis function and value indices
        tensor_i = tuple(gem.Index(extent=d) for d in self._shape)
        tensor_vi = tuple(gem.Index(extent=d) for d in self._shape)

        # Couple new basis function and value indices
        deltas = reduce(gem.Product, (gem.Delta(j, k)
                                      for j, k in zip(tensor_i, tensor_vi)))

        if self._transpose:
            index_ordering = tensor_i + scalar_i + tensor_vi + scalar_vi
        else:
            index_ordering = scalar_i + tensor_i + tensor_vi + scalar_vi

        result = {}
        for alpha, expr in scalar_evaluation.items():
            result[alpha] = gem.ComponentTensor(
                gem.Product(deltas, gem.Indexed(expr, scalar_i + scalar_vi)),
                index_ordering
            )
        return result

    @property
    def dual_basis(self):
        base_dual_basis = self._base_element.dual_basis

        tensor_dual_basis = []
        if len(self.value_shape) == 1:
            for base_dual, tensorfe_idx in base_dual_basis:
                assert tensorfe_idx is None
                for alpha in range(self.value_shape[0]):
                    tensorfe_idx = (alpha,)
                    tensor_dual_basis.append(tuple([self._tensorise_dual(base_dual, tensorfe_idx), tensorfe_idx]))
        elif len(self.value_shape) == 2:
            for base_dual, tensorfe_idx in base_dual_basis:
                assert tensorfe_idx is None
                for alpha in range(self.value_shape[0]):
                    for beta in range(self.value_shape[1]):
                        tensorfe_idx = (alpha, beta)
                        tensor_dual_basis.append(tuple([self._tensorise_dual(base_dual, tensorfe_idx), tensorfe_idx]))
        elif len(self.value_shape) == 3:
            for base_dual, tensorfe_idx in base_dual_basis:
                assert tensorfe_idx is None
                for alpha in range(self.value_shape[0]):
                    for beta in range(self.value_shape[1]):
                        for gamma in range(self.value_shape[2]):
                            tensorfe_idx = (alpha, beta, gamma)
                            tensor_dual_basis.append(tuple([self._tensorise_dual(base_dual, tensorfe_idx), tensorfe_idx]))
        else:
            raise NotImplementedError("Cannot create dual basis for rank-4 tensor-valued basis functions!")

        return tuple(tensor_dual_basis)

    def _tensorise_dual(self, base_dual, idx):
        """Tensorises given base_dual by extracting idx of expression with weight_tensor.
        """
        tensor_derivs = []
        for base_deriv in base_dual:
            tensor_pts_in_derivs = []
            for base_tups in base_deriv:
                try:
                    base_point_set, weight_tensor, alpha_tensor = base_tups
                except ValueError:  # Empty
                    tensor_pts_in_derivs.append(tuple())
                    continue

                tensor_pts_in_derivs.append((base_point_set, weight_tensor, alpha_tensor))
            tensor_derivs.append(tuple(tensor_pts_in_derivs))

        return tuple(tensor_derivs)

    @property
    def mapping(self):
        return self._base_element.mapping
