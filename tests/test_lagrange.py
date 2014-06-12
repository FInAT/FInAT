import pytest
import FIAT
import yafc


def test_build_lagrange():

    cell = FIAT.reference_element.UFCTriangle()

    lagrange = yafc.element.Lagrange(cell, 1)

    assert lagrange


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
