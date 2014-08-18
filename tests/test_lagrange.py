import pytest
import FIAT
import finat
import pymbolic.primitives as p
import numpy as np


@pytest.fixture
def lagrange():
    cell = FIAT.reference_element.UFCTriangle()

    return finat.Lagrange(cell, 1)


@pytest.fixture
def points(lagrange):
    lattice = lagrange.cell.make_lattice(1)

    return finat.PointSet(lattice)


@pytest.fixture
def coords(lagrange):

    return finat.VectorFiniteElement(lagrange, 2)


@pytest.mark.parametrize(['derivative'],
                         [[None], [finat.grad]])
def test_build_lagrange(lagrange, coords, points, derivative):

    kernel_data = finat.KernelData(coords)

    recipe = lagrange.basis_evaluation(points,
                                       kernel_data, derivative)
    sh = np.array([i.extent.stop for j in recipe.indices for i in j])

    print recipe
    assert not all(sh - kernel_data.static.values()[0][1]().shape)


def test_lagrange_field(lagrange, coords, points):

    kernel_data = finat.KernelData(coords)

    recipe = lagrange.field_evaluation(p.Variable("u"),
                                       points,
                                       kernel_data)

    print recipe.expression
    print [data() for (name, data) in kernel_data.static.values()]

    assert lagrange


def test_lagrange_moment(lagrange, coords):

    lattice = lagrange.cell.make_lattice(1)

    # trivial weights so that I don't have to wrap a quadrature rule here.
    # not hard to fix, but I want to get the rule running
    weights = finat.PointSet(np.ones((len(lattice),)))

    points = finat.PointSet(lattice)

    kernel_data = finat.KernelData(coords)

    q = finat.indices.PointIndex(3)

    v = finat.ast.Recipe(((), (), (q,)), p.Variable("f")[q])

    recipe = lagrange.moment_evaluation(v,
                                        weights,
                                        points,
                                        kernel_data)

    print recipe.expression
    print [data() for (name, data) in kernel_data.static.values()]

    assert lagrange


def test_lagrange_2form_moment(lagrange, coords):

    lattice = lagrange.cell.make_lattice(1)

    # trivial weights so that I don't have to wrap a quadrature rule here.
    # not hard to fix, but I want to get the rule running
    weights = finat.PointSet(np.ones((len(lattice),)))

    points = finat.PointSet(lattice)

    kernel_data = finat.KernelData(coords)

    trial = lagrange.basis_evaluation(points, kernel_data)

    recipe = lagrange.moment_evaluation(trial,
                                        weights,
                                        points,
                                        kernel_data)

    print recipe
    assert recipe


def test_lagrange_tabulate(lagrange, points):

    tab = lagrange._tabulate(points, None)

    assert (np.eye(3) - tab < 1.e-16).all()


def test_lagrange_tabulate_grad(lagrange, points):

    tab = lagrange._tabulate(points, finat.derivatives.grad)

    ans = np.array([[-1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0]])

    assert (tab.shape == (2, 3, 3))
    assert (np.abs(ans - tab[:, :, 0]) < 1.e-14).all()


def test_lagrange_lattice(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = finat.PointSet(lattice)

    assert (points.points == lattice).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
