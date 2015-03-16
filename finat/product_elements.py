"""Preliminary support for tensor product elements."""
from .finiteelementbase import ScalarElementMixin, FiniteElementBase
from .indices import TensorPointIndex, TensorBasisFunctionIndex, DimensionIndex
from .derivatives import grad
from .ast import Recipe, CompoundVector, IndexSum


class ScalarProductElement(ScalarElementMixin, FiniteElementBase):
    """A scalar-valued tensor product element."""
    def __init__(self, *args):
        super(ScalarProductElement, self).__init__()

        assert all([isinstance(e, FiniteElementBase) for e in args])

        self.factors = args

        self._degree = max([a._degree for a in args])
        self._cell = None

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

            # Need to replace the basisfunctionindices on phi_d with i
            expressions = [reduce(lambda a, b: a.body * b.body,
                                  phi[:d] + [phi_d[d]] + phi[d + 1:])
                           for d in range(len(phi))]

            alpha_ = [phi_.indices[0][0] for phi_ in phi_d]
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
