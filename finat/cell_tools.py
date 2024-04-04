import FIAT


def max_complex(complexes):
    """Find the maximal complex in a list of cell complexes.
    This is a pass-through from FIAT so that FInAT clients
    (e.g. tsfc) don't have to directly import FIAT."""
    return FIAT.reference_element.max_complex(complexes)
