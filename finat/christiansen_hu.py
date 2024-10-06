import FIAT

from finat.physically_mapped import Citations
from finat.piola_mapped import PiolaBubbleElement


class ChristiansenHu(PiolaBubbleElement):
    def __init__(self, cell, degree=1):
        if degree != 1:
            raise ValueError("Christiansen-Hu only defined for degree = 1")
        if Citations is not None:
            Citations().register("ChristiansenHu2019")
        super().__init__(FIAT.ChristiansenHu(cell, degree))
