import FIAT

from finat.fiat_elements import FiatElement


class NodalEnrichedElement(FiatElement):
    """An enriched element with a nodal basis."""

    def __init__(self, elements):
        fiat_elements = [elem.fiat_equivalent for elem in elements]
        nodal_enriched = FIAT.NodalEnrichedElement(*fiat_elements)
        super(NodalEnrichedElement, self).__init__(nodal_enriched)
