"""Preliminary support for quadrilateral elements. Later to be
generalised to general tensor product elements."""
from .finiteelementbase import FiniteElementBase
from .indices import TensorPointIndex

class QuadrilateralElement(FiniteElementBase):
    def __init__(self, h_element, v_element):
        super(QuadrilateralElement, self).__init__()

        self.h_element = h_element
        self.v_element = v_element

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
            raise NotImplementedError

    def field_evaluation(self, field_var, q,
                         kernel_data, derivative=None, pullback=True):
        raise NotImplementedError

    def moment_evaluation(self, value, weights, q,
                          kernel_data, derivative=None, pullback=True):
        raise NotImplementedError
