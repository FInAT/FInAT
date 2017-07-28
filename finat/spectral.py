from __future__ import absolute_import, print_function, division

import FIAT

from finat.fiat_elements import ScalarFiatElement


class GaussLobattoLegendre(ScalarFiatElement):
    """1D continuous element with nodes at the Gauss-Lobatto points."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.GaussLobattoLegendre(cell, degree)
        super(GaussLobattoLegendre, self).__init__(fiat_element)


class GaussLegendre(ScalarFiatElement):
    """1D discontinuous element with nodes at the Gauss-Legendre points."""

    def __init__(self, cell, degree):
        fiat_element = FIAT.GaussLegendre(cell, degree)
        super(GaussLegendre, self).__init__(fiat_element)
