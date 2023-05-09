from functools import reduce
from itertools import chain

import numpy

import gem
from gem.optimise import delta_elimination, sum_factorise, traverse_product
from gem.utils import cached_property

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

    @cached_property
    def _entity_dofs(self):
        dofs = {}
        base_dofs = self._base_element.entity_dofs()
        ndof = int(numpy.prod(self._shape, dtype=int))

        def expand(dofs):
            dofs = tuple(dofs)
            if self._transpose:
                space_dim = self._base_element.space_dimension()
                # Components stride by space dimension of base element
                iterable = ((v + i*space_dim for v in dofs)
                            for i in range(ndof))
            else:
                # Components packed together
                iterable = (range(v*ndof, (v+1)*ndof) for v in dofs)
            yield from chain.from_iterable(iterable)

        for dim in self.cell.get_topology().keys():
            dofs[dim] = dict((k, list(expand(d)))
                             for k, d in base_dofs[dim].items())
        return dofs

    def entity_dofs(self):
        return self._entity_dofs

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
        scalar_i = self.base_element.get_indices()
        scalar_vi = self.base_element.get_value_indices()
        tensor_i = tuple(gem.Index(extent=d) for d in self._shape)
        tensor_vi = tuple(gem.Index(extent=d) for d in self._shape)
        # Couple new basis function and value indices
        deltas = reduce(gem.Product, (gem.Delta(j, k)
                                      for j, k in zip(tensor_i, tensor_vi)))
        if self._transpose:
            index_ordering = tensor_i + scalar_i + tensor_vi + scalar_vi
        else:
            index_ordering = scalar_i + tensor_i + tensor_vi + scalar_vi

        Qi = Q[scalar_i + scalar_vi]
        tQ = gem.ComponentTensor(Qi*deltas, index_ordering)
        return tQ, points

    def dual_evaluation(self, fn):
        tQ, x = self.dual_basis
        expr = fn(x)
        # Apply targeted sum factorisation and delta elimination to
        # the expression
        sum_indices, factors = delta_elimination(*traverse_product(expr))
        expr = sum_factorise(sum_indices, factors)
        # NOTE: any shape indices in the expression are because the
        # expression is tensor valued.
        assert expr.shape == self.value_shape

        scalar_i = self.base_element.get_indices()
        scalar_vi = self.base_element.get_value_indices()
        tensor_i = tuple(gem.Index(extent=d) for d in self._shape)
        tensor_vi = tuple(gem.Index(extent=d) for d in self._shape)

        if self._transpose:
            index_ordering = tensor_i + scalar_i + tensor_vi + scalar_vi
        else:
            index_ordering = scalar_i + tensor_i + tensor_vi + scalar_vi

        tQi = tQ[index_ordering]
        expri = expr[tensor_i + scalar_vi]
        evaluation = gem.IndexSum(tQi * expri, x.indices + scalar_vi + tensor_i)
        # This doesn't work perfectly, the resulting code doesn't have
        # a minimal memory footprint, although the operation count
        # does appear to be minimal.
        evaluation = gem.optimise.contraction(evaluation)
        return evaluation, scalar_i + tensor_vi

    @property
    def mapping(self):
        return self._base_element.mapping


TensorElement = TensorFiniteElement


class VectorFiniteElement(TensorFiniteElement):
    def __init__(self, element, dim, transpose=False):
        r"""A Finite element whose basis functions have the form:

        .. math::

            \boldsymbol\phi_{i \alpha} = \mathbf{e}_{\alpha} \phi_i

        Where :math:`\{\mathbf{e}_\alpha,\, \alpha=0\ldots\mathrm{shape[0]}\}`
        is the basis for :math:`\mathbb{R}^{d}`; and
        :math:`\{\phi_i\}` is the basis for the corresponding scalar
        finite element space.

        :param element: The scalar finite element.
        :param dim: The geometric dimension (:math:`d`).
        :param transpose: Changes the DoF ordering from the
                          Firedrake-style XYZ XYZ XYZ XYZ to the
                          FEniCS-style XXXX YYYY ZZZZ.  That is,
                          tensor shape indices come before the scalar
                          basis function indices when transpose=True.

        :math:`\boldsymbol\phi_{i\alpha}` is, of course, vector-valued. If
        we subscript the vector-value with :math:`\gamma\epsilon` then we can 
        write:

        .. math::
           \boldsymbol\phi_{\gamma(i\alpha)} = \delta_{\gamma\alpha}\phi_i

        This form enables the simplification of the loop nests which
        will eventually be created, so it is the form we employ here."""
        super().__init__(element, (dim,), transpose)
