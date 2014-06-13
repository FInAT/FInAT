import pytest
import FIAT
import yafc
import pymbolic.primitives as p

@pytest.fixture
def lagrange():
    cell = FIAT.reference_element.UFCTriangle()

    return yafc.element.Lagrange(cell, 1)


def test_build_lagrange(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = yafc.element.PointSet(lattice)

    kernel_data = yafc.element.KernelData()

    recipe = lagrange.basis_evaluation(points,
                                       kernel_data)

    print recipe.instructions[0]
    print [data() for (name, data) in kernel_data.static.values()]

    assert False

    assert lagrange


def test_lagrange_field(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = yafc.element.PointSet(lattice)

    kernel_data = yafc.element.KernelData()

    recipe = lagrange.field_evaluation(p.Variable("u"),
                                       points,
                                       kernel_data)

    print recipe.instructions[0]
    print [data() for (name, data) in kernel_data.static.values()]

    assert False

    assert lagrange


def test_lagrange_lattice(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = yafc.element.PointSet(lattice)

    assert (points.points == lattice).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
