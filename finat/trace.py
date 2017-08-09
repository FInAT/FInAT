from __future__ import absolute_import, print_function, division

import FIAT

from finat.fiat_elements import ScalarFiatElement


class HDivTrace(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(HDivTrace, self).__init__(FIAT.HDivTrace(cell, degree))
