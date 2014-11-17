from finiteelementbase import FiniteElementBase


class Bernstein(FiniteElementBase):
    """Scalar-valued Bernstein element. Note: need to work out the
    correct heirarchy for different Bernstein elements."""

    def __init__(self, cell, degree):
        super(Bernstein, self).__init__()

        self._cell = cell
        self._degree = degree

    def field_evaluation(self, field_var, q, kernel_data, derivative=None):


        
