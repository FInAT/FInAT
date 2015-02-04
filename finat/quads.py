"""Preliminary support for quadrilateral elements. Later to be
generalised to general tensor product elements."""
from .finiteelementbase import ScalarElementMixin, FiniteElementBase
from .indices import TensorPointIndex, TensorBasisFunctionIndex, DimensionIndex
from .derivatives import grad
from .ast import Recipe, CompoundVector


class QuadrilateralElement(ScalarElementMixin, FiniteElementBase):
    def __init__(self, *args):
        super(QuadrilateralElement, self).__init__()

        assert all([isinstance(e, FiniteElementBase) for e in args])

        self.factors = args

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

        i_ = [phi_.indices[1] for phi_ in phi]
        i = TensorBasisFunctionIndex(*i_)

        if derivative is grad:
            raise NotImplementedError
            phi_d = [e.basis_evaluation(q_, kernel_data, grad=True)
                     for e, q_ in zip(self.factors, q.factors)]

            # Need to replace the basisfunctionindices on phi_d with i
            expressions = [reduce(lambda a, b: a.body * b.body,
                                  phi[:d] + [phi_d[d]] + phi[d + 1:])
                           for d in len(phi)]

            d_ = [phi_.indices[0] for phi_ in phi_d]
            d = DimensionIndex(sum(d__.length for d__ in d))

            expr = CompoundVector(d, d_, expressions)

            ind = ((d,), (i,), (q,))

        else:
            # note - think about pullbacks.

            # note - think about what happens in the vector case.

            ind = ((), (i,), (q,))
            expr = reduce(lambda a, b: a.body * b.body, phi)

        return Recipe(indices=ind, body=expr)
