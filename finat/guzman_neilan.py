import FIAT

from finat.physically_mapped import Citations
from finat.piola_mapped import PiolaBubbleElement


class GuzmanNeilan(PiolaBubbleElement):
    def __init__(self, cell, degree=None, subdegree=1):
        sd = cell.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError("Guzman-Neilan only defined for degree = dim")
        if Citations is not None:
            Citations().register("GuzmanNeilan2019")
        super().__init__(FIAT.GuzmanNeilan(cell, degree, subdegree=subdegree))


class GuzmanNeilanBubble(GuzmanNeilan):
    def __init__(self, cell, degree=None):
        super().__init__(cell, degree=degree, subdegree=0)
