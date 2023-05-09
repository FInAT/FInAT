import pytest

import FIAT
import finat


@pytest.fixture(params=[1, 2, 3])
def cell(request):
    dim = request.param
    return FIAT.ufc_simplex(dim)


@pytest.mark.parametrize('degree', [1, 2])
def test_lagrange_mapping_call(cell, degree):
    element = finat.Lagrange(cell, degree)
    assert isinstance(element.mapping(), str)
    assert isinstance(element.mapping, str)

    dis_cont = finat.DiscontinuousElement(element)
    assert isinstance(dis_cont.mapping(), str)
    assert isinstance(dis_cont.mapping, str)

@pytest.mark.parametrize('degree', [1, 2])
def test_enriched_mapping_call(cell, degree):
    element = finat.Lagrange(cell, degree)
    enriched = finat.EnrichedElement([element, element])
    assert isinstance(enriched.mapping(), str)
    assert isinstance(enriched.mapping, str)


def test_hermite_mapping_call(cell):
    element = finat.Hermite(cell, 3)
    assert isinstance(element.mapping(), str)
    assert isinstance(element.mapping, str)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
