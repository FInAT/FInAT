import FIAT
import finat
from finat.ast import Variable
import numpy as np

cell = FIAT.reference_element.UFCTriangle()

lagrange = finat.Lagrange(cell, 1)

vector_lagrange = finat.VectorFiniteElement(lagrange, 2)

lattice = vector_lagrange.cell.make_lattice(1)

X = Variable("X")

q = FIAT.quadrature.CollapsedQuadratureTriangleRule(cell, 2)

points = finat.indices.PointIndex(finat.PointSet(q.get_points()))

kernel_data = finat.KernelData(vector_lagrange, X, affine=False)

recipe = lagrange.basis_evaluation(points,
                                   kernel_data, pullback=False)
print recipe
print finat.interpreter.evaluate(recipe, {"bar": np.ones((4,))}, kernel_data)
