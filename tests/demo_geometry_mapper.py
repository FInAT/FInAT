import FIAT
import finat
from finat import ast
import numpy as np

cell = FIAT.reference_element.UFCTriangle()

lagrange = finat.Lagrange(cell, 1)

vector_lagrange = finat.VectorFiniteElement(lagrange, 2)

lattice = vector_lagrange.cell.make_lattice(1)

X = ast.Variable("X")

kernel_data = finat.KernelData(vector_lagrange, X, affine=False)

q = FIAT.quadrature.CollapsedQuadratureTriangleRule(cell, 2)

points = finat.indices.PointIndex(finat.PointSet(q.get_points()))

weights = finat.PointSet(q.get_weights())

var = ast.Variable("bar")

recipe = lagrange.field_evaluation(var, points,
                                   kernel_data, finat.grad, pullback=True)
recipe = vector_lagrange.moment_evaluation(recipe, weights, points,
                                           kernel_data, pullback=True)

recipe = finat.GeometryMapper(kernel_data)(recipe)
print recipe
