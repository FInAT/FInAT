import pytest
import FIAT
import finat
import pymbolic.primitives as p
import numpy as np


@pytest.fixture
def lagrange():
    cell = FIAT.reference_element.UFCTriangle()

    return finat.Lagrange(cell, 1)


def test_build_lagrange(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = finat.PointSet(lattice)

    kernel_data = finat.KernelData()

    recipe = lagrange.basis_evaluation(points,
                                       kernel_data)

    print recipe.instructions[0]
    print [data() for (name, data) in kernel_data.static.values()]
    print

    assert False

    assert lagrange


def test_lagrange_field(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = finat.PointSet(lattice)

    kernel_data = finat.KernelData()

    recipe = lagrange.field_evaluation(p.Variable("u"),
                                       points,
                                       kernel_data)

    print recipe.instructions[0]
    print [data() for (name, data) in kernel_data.static.values()]

    assert False

    assert lagrange


def test_lagrange_moment(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    # trivial weights so that I don't have to wrap a quadrature rule here.
    # not hard to fix, but I want to get the rule running
    weights = finat.PointSet(np.ones((len(lattice),)))

    points = finat.PointSet(lattice)

    kernel_data = finat.KernelData()

    recipe = lagrange.moment_evaluation(p.Variable("f"),
                                        weights,
                                        points,
                                        kernel_data)

    print recipe.instructions[0]
    print [data() for (name, data) in kernel_data.static.values()]

    assert False

    assert lagrange


def test_lagrange_lattice(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = finat.PointSet(lattice)

    assert (points.points == lattice).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
