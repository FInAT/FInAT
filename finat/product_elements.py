"""Preliminary support for tensor product elements."""
from .finiteelementbase import ScalarElementMixin, FiniteElementBase
from .indices import TensorPointIndex, TensorBasisFunctionIndex, DimensionIndex
from .derivatives import grad
from .ast import Recipe, CompoundVector, IndexSum
from FIAT.reference_element import two_product_cell

class ProductElement(object):
    """Mixin class describing product elements."""

class ScalarProductElement(ProductElement, ScalarElementMixin, FiniteElementBase):
    """A scalar-valued tensor product element."""
    def __init__(self, *args):
        super(ScalarProductElement, self).__init__()

        assert all([isinstance(e, FiniteElementBase) for e in args])

        self.factors = tuple(args)

        self._degree = max([a._degree for a in args])

        cellprod = lambda cells: two_product_cell(cells[0], cells[1] if len(cells) < 3
                                                  else cellprod(cells[1:]))

        self._cell = cellprod([a.cell for a in args])

    def basis_evaluation(self, q, kernel_data, derivative=None,
                         pullback=True):
        '''Produce the variable for the tabulation of the basis
        functions or their derivative.'''

        assert isinstance(q, TensorPointIndex)

        if derivative not in (None, grad):
            raise ValueError(
                "Scalar elements do not have a %s operation") % derivative

        phi = [e.basis_evaluation(q_, kernel_data)
               for e, q_ in zip(self.factors, q.factors)]

        i_ = [phi_.indices[1][0] for phi_ in phi]
        i = TensorBasisFunctionIndex(*i_)

        if derivative is grad:

            phi_d = [e.basis_evaluation(q_, kernel_data, derivative=grad, pullback=False)
                     for e, q_ in zip(self.factors, q.factors)]

            # Replace the basisfunctionindices on phi_d with i
            phi_d = [p.replace_indices(zip(p.indices[1], (i__,)))
                     for p, i__ in zip(phi_d, i_)]

            expressions = tuple(reduce(lambda a, b: a.body * b.body,
                                       phi[:d] + [phi_d[d]] + phi[d + 1:])
                                for d in range(len(phi)))

            alpha_ = tuple(phi_.indices[0][0] for phi_ in phi_d)
            alpha = DimensionIndex(sum(alpha__.length for alpha__ in alpha_))

            assert alpha.length == kernel_data.gdim
            expr = CompoundVector(alpha, alpha_, expressions)

            if pullback:
                beta = alpha
                alpha = DimensionIndex(kernel_data.gdim)
                invJ = kernel_data.invJ[(beta, alpha)]
                expr = IndexSum((beta,), invJ * expr)

            ind = ((alpha,), (i,), (q,))

        else:

            ind = ((), (i,), (q,))
            expr = reduce(lambda a, b: a.body * b.body, phi)

        return Recipe(indices=ind, body=expr)

    def __hash__(self):
        """ScalarProductElements are equal if their factors are equal"""

        return hash(self.factors)

    def __eq__(self, other):
        """VectorFiniteElements are equal if they have the same base element
        and dimension."""

        return self.factors == other.factors
