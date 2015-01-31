import FIAT
import finat
import pymbolic.primitives as p
import numpy as np

cell = FIAT.reference_element.UFCTriangle()

lagrange = finat.Lagrange(cell, 1)

vector_lagrange = finat.VectorFiniteElement(lagrange, 2)

lattice = vector_lagrange.cell.make_lattice(1)

X = p.Variable("X")

q = FIAT.quadrature.CollapsedQuadratureTriangleRule(cell, 2)

points = finat.indices.PointIndex(finat.PointSet(q.get_points()))

kernel_data = finat.KernelData(vector_lagrange, X, affine=False)

recipe = lagrange.basis_evaluation(points,
                                   kernel_data, pullback=False)

# print recipe
gvm = finat.ast.GraphvizMapper()
gvm(recipe)
print gvm.get_dot_code()
