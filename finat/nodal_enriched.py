import FIAT

from finat.fiat_elements import FiatElement


class NodalEnrichedElement(FiatElement):
    """An enriched element with a nodal basis."""

    def __init__(self, elements):
        nodal_enriched = FIAT.NodalEnrichedElement(*(elem.fiat_equivalent
                                                     for elem in elements))
        super().__init__(nodal_enriched)
