import FIAT

from finat.fiat_elements import FiatElement
from finat.piola_mapped import PiolaMappedElement


class Stokes(PiolaMappedElement):
    """Pk^d"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.Stokes(cell, degree=degree))


class MacroStokes(PiolaMappedElement):
    """C0 Pk^d(Alfeld)"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.MacroStokes(cell, degree=degree))


class DivStokes(FiatElement):
    """Pk"""
    def __init__(self, cell, degree=None):
        super().__init__(FIAT.DivStokes(cell, degree=degree))
