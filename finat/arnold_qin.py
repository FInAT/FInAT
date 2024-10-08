import FIAT

from finat.physically_mapped import Citations
from finat.fiat_elements import FiatElement
from finat.piola_mapped import PiolaBubbleElement


class ArnoldQin(FiatElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree))


class ReducedArnoldQin(PiolaBubbleElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("ArnoldQin1992")
        super().__init__(FIAT.ArnoldQin(cell, degree, reduced=True))
