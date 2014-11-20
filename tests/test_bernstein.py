import pytest
import FIAT
import finat
import pymbolic.primitives as p
import numpy as np


@pytest.fixture
def cell():
    return FIAT.reference_element.UFCTriangle()


@pytest.fixture
def lagrange(cell):

    return finat.Lagrange(cell, 1)


@pytest.fixture
def coords(lagrange):

    return finat.VectorFiniteElement(lagrange, 2)


@pytest.fixture
def quadrature(cell):
    return finat.quadrature.StroudQuadrature(cell, 2)


@pytest.fixture
def bernstein(cell):
    return finat.Bernstein(cell, 1)


def test_bernstein_field(coords, quadrature, bernstein):

    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)

    recipe = bernstein.field_evaluation(p.Variable("u"),
                                        q,
                                        kernel_data)
    print recipe
    assert False


#kernel_data = finat.KernelData(finat.VectorFiniteElement(lagrange(cell()), 2))

#q = finat.indices.TensorPointIndex(quadrature(cell()).points)

#bernstein = bernstein(cell())

#recipe = bernstein.field_evaluation(p.Variable("u"),
#                                    q, kernel_data)
#print bernstein.dofs_shape
#udata = np.ones(bernstein.dofs_shape)
#finat.interpreter.evaluate(recipe, context={"u": udata},
#                           kernel_data=kernel_data)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
