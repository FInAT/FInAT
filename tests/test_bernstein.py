import pytest
import FIAT
import finat
from finat.ast import Variable
import numpy as np


@pytest.fixture
def cell():

    return FIAT.reference_element.UFCTriangle()


@pytest.fixture
def lagrange(cell):

    return finat.Lagrange(cell, 3)


@pytest.fixture
def coords(lagrange):

    return finat.VectorFiniteElement(lagrange, 2)


@pytest.fixture
def quadrature(cell):

    return finat.quadrature.StroudQuadrature(cell, 8)


@pytest.fixture
def bernstein(cell):

    return finat.Bernstein(cell, 1)


def test_bernstein_field(coords, quadrature, bernstein):

    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)

    recipe = bernstein.field_evaluation(Variable("u"),
                                        q,
                                        kernel_data)
    print recipe
    assert True


def test_bernstein_moment(coords, quadrature, bernstein):

    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)
    wt = quadrature.weights

    recipe = bernstein.moment_evaluation(Variable("f"),
                                         wt,
                                         q,
                                         kernel_data)
    print recipe
    assert True


def test_interpret_bernstein_field(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(finat.VectorFiniteElement(lagrange(cell()), 2))

    q = finat.indices.TensorPointIndex(quadrature.points)

    recipe = bernstein.field_evaluation(Variable("u"),
                                        q, kernel_data)

    udata = np.ones(bernstein.dofs_shape)
    print finat.interpreter.evaluate(recipe, context={"u": udata},
                                     kernel_data=kernel_data)

    assert True


@pytest.mark.xfail
def test_interpret_bernstein_moment(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(finat.VectorFiniteElement(lagrange(cell()), 2))

    q = finat.indices.TensorPointIndex(quadrature.points)
    wt = quadrature.weights

    recipe = bernstein.moment_evaluation(Variable("f"),
                                         wt,
                                         q,
                                         kernel_data)

    nqp1d = len(wt[0])
    fdata = np.ones((nqp1d, nqp1d))
    print finat.interpreter.evaluate(recipe, context={"f": fdata},
                                     kernel_data=kernel_data)

    assert False

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
