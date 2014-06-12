import pytest
import FIAT
import yafc

@pytest.fixture
def lagrange():
    cell = FIAT.reference_element.UFCTriangle()

    return yafc.element.Lagrange(cell, 1)


def test_build_lagrange(lagrange):

    assert lagrange


def test_lagrange_lattice(lagrange):

    lattice = lagrange.cell.make_lattice(1)

    points = yafc.element.PointSet(lattice)

    assert points.points == lattice


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
