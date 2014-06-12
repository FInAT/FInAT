from finiteelementbase import FiniteElementBase

import FIAT

class Lagrange(FiniteElementBase):
    def __init__(self, cell, degree):

        self._cell = cell
        self._degree = degree

        self._fiat_element = FIAT.Lagrange(cell, degree)
