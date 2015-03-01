import pytest
import FIAT
import finat


@pytest.fixture
def cell():
    return FIAT.reference_element.UFCTriangle()


def quadrature(cell, degree):
    q = FIAT.quadrature.CollapsedQuadratureTriangleRule(cell, 2)

    points = finat.indices.PointIndex(finat.PointSet(q.get_points()))

    weights = finat.PointSet(q.get_weights())

    return(points, weights)


@pytest.mark.parametrize(['degree'],
                         [[i] for i in range(1, 6)])
def test_integrate_cell(cell, degree):

    # This is overkill, but it exercises some code.
    q, w = quadrature(cell, degree)

    l = finat.Lagrange(cell, degree)

    vl = finat.VectorFiniteElement(l, 2)

    kernel_data = finat.KernelData(vl)

    recipe = l.moment_evaluation(
        finat.ast.Recipe(((), (), ()), 1.), w, q, kernel_data, pullback=False)

    assert abs(finat.interpreter.evaluate(recipe, {}, kernel_data).sum() - 0.5) < 1.e-12


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
