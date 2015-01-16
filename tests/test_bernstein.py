import pytest
import FIAT
import finat
import pymbolic.primitives as p
import numpy as np


@pytest.fixture
def cell():
    return FIAT.reference_element.UFCTriangle()
    #return FIAT.reference_element.UFCInterval()


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

    recipe = bernstein.field_evaluation(p.Variable("u"),
                                        q,
                                        kernel_data)
    print recipe
    assert False

def test_bernstein_moment(coords, quadrature, bernstein):

    kernel_data = finat.KernelData(coords)

    q = finat.indices.TensorPointIndex(quadrature.points)
    wt = quadrature.weights

    recipe = bernstein.moment_evaluation(p.Variable("f"),
                                         wt,
                                         q,
                                         kernel_data)
    print recipe
    assert False


def test_interpret_bernstein_field(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(finat.VectorFiniteElement(lagrange(cell()), 2))

    q = finat.indices.TensorPointIndex(quadrature.points)

    recipe = bernstein.field_evaluation(p.Variable("u"),
                                        q, kernel_data)

    udata = np.ones(bernstein.dofs_shape)
    print finat.interpreter.evaluate(recipe, context={"u": udata},
                                     kernel_data=kernel_data)

    assert False

def test_interpret_bernstein_moment(coords, quadrature, bernstein):
    kernel_data = finat.KernelData(finat.VectorFiniteElement(lagrange(cell()), 2))

    q = finat.indices.TensorPointIndex(quadrature.points)
    wt = quadrature.weights

    recipe = bernstein.moment_evaluation(p.Variable("f"),
                                         wt,
                                         q,
                                         kernel_data)

    nqp = len(wt[0])**2
    fdata = np.ones((nqp,))
    print finat.interpreter.evaluate(recipe, context={"f": fdata},
                                     kernel_data=kernel_data)

    assert False

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
