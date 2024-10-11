import FIAT

from finat.physically_mapped import Citations
from finat.piola_mapped import PiolaBubbleElement


class GuzmanNeilan(PiolaBubbleElement):
    """Pk^d enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, order=1):
        if Citations is not None:
            Citations().register("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilan(cell, order=order))


class GuzmanNeilanSecondKind(PiolaBubbleElement):
    """C0 Pk^d(Alfeld) enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, order=1):
        if Citations is not None:
            Citations().register("GuzmanNeilan2018")
        super().__init__(FIAT.GuzmanNeilanSecondKind(cell, order=order))


class GuzmanNeilanBubble(GuzmanNeilan):
    """Modified Bernardi-Raugel bubbles that are C^0 P_dim(Alfeld) with constant divergence."""
    def __init__(self, cell, degree=None):
        super().__init__(cell, order=0)


class GuzmanNeilanH1div(PiolaBubbleElement):
    """Alfeld-Sorokina nodally enriched with Guzman-Neilan bubbles."""
    def __init__(self, cell, degree=None):
        if Citations is not None:
            Citations().register("GuzmanNeilan2018")
        AS = FIAT.AlfeldSorokina(cell, 2)
        GNBubble = FIAT.GuzmanNeilan(cell, order=0)
        super().__init__(FIAT.NodalEnrichedElement(AS, GNBubble))
