"""Preliminary support for quadrilateral elements. Later to be
generalised to general tensor product elements."""
from .finiteelementbase import ScalarElementMixin, FiniteElementBase
from .indices import TensorPointIndex, TensorBasisFunctionIndex
from .derivatives import grad
from .ast import Recipe


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

        if derivative is grad:
            raise NotImplementedError

        else:
            # note - think about pullbacks.
            phi = [e.basis_evaluation(q_, kernel_data)
                   for e, q_ in zip(self.factors, q.factors)]

            # note - think about what happens in the vector case.
            i_ = [phi_.indices[1] for phi_ in phi]
            i = TensorBasisFunctionIndex(*i_)

            ind = ((), (i,), (q,))
            expr = reduce(lambda a, b: a.body * b.body, phi)

        return Recipe(indices=ind, body=expr)
