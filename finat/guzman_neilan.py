import FIAT

from finat.physically_mapped import Citations
from finat.piola_mapped import PiolaBubbleElement


class GuzmanNeilan(PiolaBubbleElement):
    """H1-conforming macroelement construced as the enrichment of Pk^d
    with Guzman-Neilan facet bubbles.
    """
    def __init__(self, cell, degree=None, subdegree=1):
        if Citations is not None:
            Citations().register("GuzmanNeilan2019")
        super().__init__(FIAT.GuzmanNeilan(cell, degree=degree, subdegree=subdegree))


class GuzmanNeilanBubble(GuzmanNeilan):
    """Normal facet bubbles that are C^0 P_dim on the Alfeld split
    with constant divergence.
    """
    def __init__(self, cell, degree=None):
        super().__init__(cell, degree=degree, subdegree=0)


class GuzmanNeilanH1div(PiolaBubbleElement):
    """H1(div)-conforming macroelement constructed as the nodal enrichement of
    Alfeld-Sorokina with Guzman-Neilan facet bubbles.
    """
    def __init__(self, cell, degree=None):
        if Citations is not None:
            Citations().register("GuzmanNeilan2019")
        AS = FIAT.AlfeldSorokina(cell, 2)
        GNBubble = FIAT.GuzmanNeilan(cell, degree=degree, subdegree=0)
        super().__init__(FIAT.NodalEnrichedElement(AS, GNBubble))
