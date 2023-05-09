import pytest

import FIAT
import finat


@pytest.fixture(params=[1, 2])
def cell(request):
    dim = request.param
    return FIAT.ufc_simplex(dim)


def test_enriched_elem(cell):
    # Test algebra on finat elements results in the correct finat elements

    S = finat.Lagrange(cell, 1)
    V = finat.VectorElement(S, 1)
    T = finat.TensorFiniteElement(S, (2,2))

    M = T * S
    assert(isinstance(M, finat.EnrichedElement))

    M = V * (S * T)
    assert(isinstance(M, finat.EnrichedElement))

    M = S + S
    assert(isinstance(M, finat.EnrichedElement))

    M = T + T
    assert(isinstance(M, finat.EnrichedElement))

def test_restricted_elem(cell):
    # Test restriction completes
    S = finat.Lagrange(cell, 3)

    M = S['interior']
    from finat.fiat_elements import FiatElement
    assert(isinstance(M, FiatElement))

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
