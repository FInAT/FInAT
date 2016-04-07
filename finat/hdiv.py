from .fiat_elements import FiatElement
import FIAT


class HDivElement(FiatElement):
    pass


class RaviartThomas(HDivElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.RaviartThomas(cell, degree)


class BrezziDouglasMarini(HDivElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasMarini(cell, degree)


class BrezziDouglasFortinMarini(HDivElement):
    def __init__(self, cell, degree):
        super(RaviartThomas, self).__init__(cell, degree)

        self._fiat_element = FIAT.BrezziDouglasFortinMarini(cell, degree)
