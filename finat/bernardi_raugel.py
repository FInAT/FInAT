import FIAT

from finat.physically_mapped import Citations
from finat.piola_mapped import PiolaBubbleElement


class BernardiRaugel(PiolaBubbleElement):
    def __init__(self, cell, degree=None, subdegree=1):
        sd = cell.get_spatial_dimension()
        if degree is None:
            degree = sd
        if Citations is not None:
            Citations().register("BernardiRaugel1985")
        super().__init__(FIAT.BernardiRaugel(cell, degree, subdegree=subdegree))


class BernardiRaugelBubble(BernardiRaugel):
    def __init__(self, cell, degree=None):
        super().__init__(cell, degree=degree, subdegree=0)
