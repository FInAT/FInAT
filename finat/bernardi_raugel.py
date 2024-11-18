import FIAT

from finat.physically_mapped import Citations
from finat.piola_mapped import PiolaBubbleElement


class BernardiRaugel(PiolaBubbleElement):
    def __init__(self, cell, order=1):
        if Citations is not None:
            Citations().register("BernardiRaugel1985")
        super().__init__(FIAT.BernardiRaugel(cell, order=order))


class BernardiRaugelBubble(BernardiRaugel):
    def __init__(self, cell, degree=None):
        super().__init__(cell, order=0)
