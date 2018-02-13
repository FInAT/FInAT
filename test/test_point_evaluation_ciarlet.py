import pytest

import FIAT
import finat
import gem


@pytest.fixture(params=[1, 2, 3])
def cell(request):
    dim = request.param
    return FIAT.ufc_simplex(dim)


@pytest.mark.parametrize('degree', [1, 2])
def test_cellwise_constant(cell, degree):
    dim = cell.get_spatial_dimension()
    element = finat.Lagrange(cell, degree)
    index = gem.Index()
    point = gem.partial_indexed(gem.Variable('X', (17, dim)), (index,))

    order = 2
    for alpha, table in element.point_evaluation(order, point).items():
        if sum(alpha) < degree:
            assert table.free_indices == (index,)
        else:
            assert table.free_indices == ()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
