import FIAT

from finat.physically_mapped import Citations
from finat.piola_mapped import PiolaBubbleElement


class GuzmanNeilan(PiolaBubbleElement):
    """H1-conforming macroelement construced as the enrichment of Pk^d
    with Guzman-Neilan facet bubbles.
    """
    def __init__(self, cell, order=1):
        if Citations is not None:
            Citations().register("GuzmanNeilan2019")
        super().__init__(FIAT.GuzmanNeilan(cell, order=order))


class GuzmanNeilanSecondKind(PiolaBubbleElement):
    """H1-conforming macroelement construced as the enrichment of C0 Pk^d(Alfeld)
    with Guzman-Neilan facet bubbles.
    """
    def __init__(self, cell, order=1):
        if Citations is not None:
            Citations().register("GuzmanNeilan2019")
        super().__init__(FIAT.GuzmanNeilanSecondKind(cell, order=order))


class GuzmanNeilanBubble(GuzmanNeilan):
    """Normal facet bubbles that are C^0 P_dim on the Alfeld split
    with constant divergence.
    """
    def __init__(self, cell, degree=None):
        super().__init__(cell, order=0)


class GuzmanNeilanH1div(PiolaBubbleElement):
    """H1(div)-conforming macroelement constructed as the nodal enrichement of
    Alfeld-Sorokina with Guzman-Neilan facet bubbles.
    """
    def __init__(self, cell, degree=None):
        if Citations is not None:
            Citations().register("GuzmanNeilan2019")
        AS = FIAT.AlfeldSorokina(cell, 2)
        GNBubble = FIAT.GuzmanNeilan(cell, order=0)
        super().__init__(FIAT.NodalEnrichedElement(AS, GNBubble))
