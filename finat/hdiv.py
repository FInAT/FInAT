import pymbolic.primitives as p
from finiteelementbase import FiatElement
from ast import Recipe, IndexSum
import FIAT
import indices
from derivatives import div, grad, curl
import numpy as np


class HDivElement(FiatElement):
    def __init__(self, cell, degree):
        super(HDivElement, self).__init__(cell, degree)

    

class RaviartThomas(HDivElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.RaviartThomas(cell, degree)
