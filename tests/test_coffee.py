import pytest
import FIAT
import finat
import numpy as np
from finat.ast import Variable


fiat_cell = {1: FIAT.reference_element.UFCInterval(),
             2: FIAT.reference_element.UFCTriangle(),
             3: FIAT.reference_element.UFCTetrahedron()}


fiat_rule = {1: FIAT.quadrature.GaussJacobiQuadratureLineRule,
             2: FIAT.quadrature.CollapsedQuadratureTriangleRule,
             3: FIAT.quadrature.CollapsedQuadratureTetrahedronRule}


@pytest.fixture
def cell(dim):
    return fiat_cell[dim]


@pytest.fixture
def lagrange(cell):
    return finat.Lagrange(cell, 1)


@pytest.fixture
def kernel_data(lagrange, dim):
    vector_lagrange = finat.VectorFiniteElement(lagrange, dim)

    return finat.KernelData(vector_lagrange, Variable("X"))


@pytest.fixture
def context(cell):
    return {'u': np.array([0.0, 0.6, 0.3, 0.4]),
            'X': np.array(cell.make_lattice(1))}


@pytest.fixture
def quadrature(cell, dim, degree):
    q = fiat_rule[dim](cell, degree)

    points = finat.indices.PointIndex(finat.PointSet(q.get_points()))

    weights = finat.PointSet(q.get_weights())

    return points, weights


# @pytest.mark.skip
# @pytest.mark.parametrize('dim', [1, 2, 3])
# @pytest.mark.parametrize('degree', [1, 2, 3])
# def test_basis_evaluation(cell, lagrange, quadrature, kernel_data, degree):
#     points, weights = quadrature

#     recipe = lagrange.basis_evaluation(points, kernel_data, derivative=None)

#     result_finat = finat.interpreter.evaluate(recipe, {}, kernel_data)

#     result_coffee = finat.coffee_compiler.evaluate(recipe, {}, kernel_data)

#     assert(np.abs(result_finat - result_coffee) < 1.e-12).all()


# @pytest.mark.skip
# @pytest.mark.parametrize('dim', [1, 2, 3])
# @pytest.mark.parametrize('degree', [1, 2, 3])
# def test_field_evaluation(cell, lagrange, quadrature, kernel_data, context, degree):
#     points, weights = quadrature

#     recipe = lagrange.field_evaluation(Variable("u"), points,
#                                        kernel_data, derivative=None)

#     result_finat = finat.interpreter.evaluate(recipe, context, kernel_data)

#     result_coffee = finat.coffee_compiler.evaluate(recipe, context, kernel_data)

#     assert(np.abs(result_finat - result_coffee) < 1.e-12).all()


# @pytest.mark.skip
# @pytest.mark.parametrize('dim', [1, 2, 3])
# @pytest.mark.parametrize('degree', [1, 2, 3])
# def test_moment_evaluation(cell, lagrange, quadrature, kernel_data, context, degree):
#     points, weights = quadrature

#     f_recipe = lagrange.field_evaluation(Variable("u"), points,
#                                          kernel_data, derivative=None)
#     recipe = lagrange.moment_evaluation(f_recipe, weights, points,
#                                         kernel_data, pullback=False)

#     result_finat = finat.interpreter.evaluate(recipe, context, kernel_data)

#     result_coffee = finat.coffee_compiler.evaluate(recipe, context, kernel_data)

#     assert(np.abs(result_finat - result_coffee) < 1.e-12).all()


# @pytest.mark.skip
# @pytest.mark.parametrize('dim', [2, 3])
# @pytest.mark.parametrize('degree', [1, 2, 3])
# def test_grad_evaluation(cell, lagrange, quadrature, kernel_data, context, degree):
#     points, weights = quadrature

#     f_recipe = lagrange.field_evaluation(Variable("u"), points, kernel_data,
#                                          derivative=finat.grad, pullback=True)
#     recipe = lagrange.moment_evaluation(f_recipe, weights, points, kernel_data,
#                                         derivative=finat.grad, pullback=True)

#     recipe = finat.GeometryMapper(kernel_data)(recipe)

#     result_finat = finat.interpreter.evaluate(recipe, context, kernel_data)

#     result_coffee = finat.coffee_compiler.evaluate(recipe, context, kernel_data)

#     assert(np.abs(result_finat - result_coffee) < 1.e-12).all()
