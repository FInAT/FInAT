import FIAT

from finat.fiat_elements import ScalarFiatElement


class HDivTrace(ScalarFiatElement):
    def __init__(self, cell, degree):
        super().__init__(FIAT.HDivTrace(cell, degree))
