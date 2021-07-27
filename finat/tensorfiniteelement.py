from functools import reduce

import numpy

import gem

from finat.finiteelementbase import FiniteElementBase
from finat.tensor_product import TensorProductElement


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

        base = self.base_element
        Q, points = base.dual_basis

        # Suppose the tensor element has shape (2, 4)
        # These identity matrices may have difference sizes depending the shapes
        # tQ = Q ⊗ 𝟙₂ ⊗ 𝟙₄
        Q_shape_indices = gem.indices(len(Q.shape))
        i_s = tuple(gem.Index(extent=d) for d in self._shape)
        j_s = tuple(gem.Index(extent=d) for d in self._shape)
        # we need one delta for each shape axis
        deltas = reduce(gem.Product, (gem.Delta(i, j) for i, j in zip(i_s, j_s)))
        # TODO Need to check how this plays with the transpose argument to TensorFiniteElement.
        tQ = gem.ComponentTensor(Q[Q_shape_indices]*deltas, Q_shape_indices + i_s + j_s)

        return tQ, points

    def dual_evaluation(self, fn):

        tQ, x = self.dual_basis

        expr = fn(x)

        #
        # TENSOR CONTRACT Q WITH expr
        #

        # NOTE: any shape indices in the expression are because the expression
        # is tensor valued.
        assert set(expr.shape) == set(self.value_shape)
        # TODO: Add shortcut (if relevant) for tQ being identity tensor
        # TODO: generalise to general rank shape and expression indices
        base_value_indices = self.base_element.get_value_indices()
        Q, _ = self.base_element.dual_basis
        Q_shape_indices = tuple(gem.Index(extent=ex) for ex in Q.shape)
        delta_indices_1 = tuple(gem.Index(extent=d) for d in self._shape)
        delta_indices_2 = tuple(gem.Index(extent=d) for d in self._shape)
        basis_indices = Q_shape_indices[:1] + delta_indices_2
        expr_indexed = expr[base_value_indices + delta_indices_1]
        # TODO: what if the basis indices in the shape of Q are more than just
        # the first shape index as in tensor product elements?
        tQ_indexed = tQ[basis_indices + base_value_indices + delta_indices_1]
        dual_evaluation_indexed_sum = gem.optimise.make_product((tQ_indexed, expr_indexed), x.indices + base_value_indices + delta_indices_1)
        if len(base_value_indices) != 0:
            raise NotImplementedError('Cannot dual evaluate tensor-valued non-scalar elements yet')
        if isinstance(self.base_element, TensorProductElement):
            raise NotImplementedError('Cannot dual evaluate tensor-valued tensor product element yet')
        return dual_evaluation_indexed_sum, basis_indices

    @property
    def mapping(self):
        return self._base_element.mapping
