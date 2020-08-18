import numpy

from finat.finiteelementbase import FiniteElementBase
from finat.physically_mapped import DirectlyDefinedElement, Citations
from FIAT.reference_element import UFCTriangle, UFCTetrahedron
from FIAT.polynomial_set import mis

from itertools import chain
import gem
import sympy
from finat.sympy2gem import sympy2gem


class ModifiedBR(DirectlyDefinedElement, FiniteElementBase):
    def __init__(self, cell, degree):
        assert isinstance(cell, (UFCTriangle, UFCTetrahedron))
        if degree != 1:
            raise NotImplementedError("Oh no!")
        self._cell = cell
        self._degree = degree

    @property
    def cell(self):
        return self._cell

    @property
    def degree(self):
        return self._degree

    @property
    def formdegree(self):
        return self.cell.get_spatial_dimension() - 1

    def entity_dofs(self):
        if self.cell.get_spatial_dimension() == 2:
            return {0: {i: [i, i+3] for i in range(3)},
                    1: {i: [] for i in range(3)},
                    # 1: {i: [6 + i] for i in range(3)},
                    2: {0: []}}
        elif self.cell.get_spatial_dimension() == 3:
            return {0: {i: [i, i + 4, i + 8] for i in range(4)},
                    1: {i: [] for i in range(6)},
                    2: {i: [] for i in range(4)},
                    # 2: {i: [12 + i] for i in range(4)},
                    3: {0: []}}
        else:
            raise NotImplementedError("Oops")

    def space_dimension(self):
        if self.cell.get_spatial_dimension() == 2:
            return 6
            # return 9
        elif self.cell.get_spatial_dimension() == 3:
            return 8
            # return 16
        else:
            raise NotImplementedError("Oh no")

    @property
    def index_shape(self):
        return (self.space_dimension(), )

    @property
    def value_shape(self):
        return (self.cell.get_spatial_dimension(), )

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        # Goal
        # Construct sympy expressions for the basis functions in terms
        # of the physical locations of the quad points.

        dim = self.cell.get_spatial_dimension()

        if dim == 2:
            phys_cell_vertices = numpy.asarray(sympy.symbols('x:3,y:3')).reshape(2, 3).T
        elif dim == 3:
            phys_cell_vertices = numpy.asarray(sympy.symbols('x:4,y:4,z:4')).reshape(3, 4).T

        gem_cell_vertices = coordinate_mapping.physical_vertices()

        phys_quad_points = sympy.symbols('x,y,z')[:dim]
        gem_quad_points = gem.partial_indexed(
            coordinate_mapping.physical_points(ps, entity=entity),
            ps.indices)

        bindings = dict(zip(phys_quad_points, gem_quad_points))

        assert phys_cell_vertices.shape == gem_cell_vertices.shape
        bindings.update((phys_cell_vertices[idx], gem_cell_vertices[idx])
                        for idx in numpy.ndindex(phys_cell_vertices.shape))

        mapper = gem.node.Memoizer(sympy2gem)
        mapper.bindings = bindings
        phis = basis_evaluation(phys_cell_vertices, phys_quad_points,
                                self.cell, self.degree)

        result = {}
        for i in range(order+1):
            alphas = mis(2, i)
            for alpha in alphas:
                dphis = [phi.diff(*tuple(zip(phys_quad_points, alpha)))
                         for phi in phis]
                result[alpha] = gem.ListTensor(list(map(mapper, dphis)))
        return result

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError("Not done yet, sorry!")

    def mapping(self):
        return "physical"


def basis_evaluation(cell_vertices, quad_points, cell, degree):
    assert degree == 1
    # First the vertex dofs
    lagrange = lagrange_basis(cell_vertices, quad_points, cell, degree)
    # vector-expand
    basis = ([sympy.Array([b, 0]) for b in lagrange]
             + [sympy.Array([0, b]) for b in lagrange])
    return basis


def lagrange_basis(cell_vertices, quad_points, cell, degree):
    assert degree == 1
    assert cell.get_spatial_dimension() == 2
    x, y = quad_points

    basis = [1 + x*0, x, y]

    dof1 = lambda phi: phi.subs({x: cell_vertices[0][0],
                                 y: cell_vertices[0][1]})
    dof2 = lambda phi: phi.subs({x: cell_vertices[1][0],
                                 y: cell_vertices[1][1]})
    dof3 = lambda phi: phi.subs({x: cell_vertices[2][0],
                                 y: cell_vertices[2][1]})

    A = [[dof1(basis[0]), dof1(basis[1]), dof1(basis[2])],
         [dof2(basis[0]), dof2(basis[1]), dof2(basis[2])],
         [dof3(basis[0]), dof3(basis[1]), dof3(basis[2])]]

    AT = sympy.Matrix(A).T
    coefficients = AT.inv('LU')
    return coefficients * sympy.Matrix(basis)
