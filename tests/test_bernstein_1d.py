import pytest
import FIAT
import finat
from finat.ast import Variable
from finat.derivatives import grad
import numpy as np


@pytest.fixture
def cell():

    return FIAT.reference_element.UFCInterval()


@pytest.fixture
def lagrange(cell):

    return finat.Lagrange(cell, 3)


@pytest.fixture
def coords(lagrange):

    return finat.VectorFiniteElement(lagrange, 1)


@pytest.fixture
def quadrature(cell):

    return finat.quadrature.StroudQuadrature(cell, 8)


@pytest.fixture
def bernstein(cell):

    return finat.Bernstein1D(cell, 1)


def test_bernstein_field(coords, quadrature, bernstein):

    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)

    recipe = bernstein.field_evaluation(Variable("u"),
                                        q,
                                        kernel_data)
    print recipe


def test_bernstein_gradient(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(coords)

    recipe = bernstein.gradient(Variable("u"),
                                kernel_data)

    print recipe
#    assert False


def test_bernstein_field_gradient(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)

    recipe = bernstein.field_evaluation(Variable("u"),
                                        q,
                                        kernel_data,
                                        derivative=grad)

    print recipe


def test_bernstein_interpret_gradient(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(coords)

    recipe = bernstein.gradient(Variable("u"),
                                kernel_data)

    #    udata = np.ones(bernstein.dofs_shape)
    udata = np.arange(2)
    vals = finat.interpreter.evaluate(recipe, context={"u": udata},
                                      kernel_data=kernel_data)
    print vals
    assert False


def test_bernstein_interpret_field(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)
    recipe = bernstein.field_evaluation(Variable("u"),
                                        q, kernel_data)

    udata = np.ones(bernstein.dofs_shape)
    vals = finat.interpreter.evaluate(recipe, context={"u": udata},
                                      kernel_data=kernel_data)

    assert np.allclose(vals, np.ones(vals.shape))


def test_bernstein_interpret_field_gradient(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)

    recipe = bernstein.field_evaluation(Variable("u"),
                                        q, kernel_data,
                                        derivative=grad)

    udata = np.ones(bernstein.dofs_shape)
    vals = finat.interpreter.evaluate(recipe, context={"u": udata},
                                      kernel_data=kernel_data)

    assert np.allclose(vals, np.zeros(vals.shape))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
