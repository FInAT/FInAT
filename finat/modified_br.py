from itertools import combinations, count
from operator import methodcaller

import gem
import numpy
import numpy as np
import sympy as sp
from FIAT.polynomial_set import mis
from FIAT.reference_element import UFCTetrahedron, UFCTriangle, ufc_simplex

from finat.finiteelementbase import FiniteElementBase
from finat.physically_mapped import Citations, DirectlyDefinedElement
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
            phys_cell_vertices = numpy.asarray(sp.symbols('x:3,y:3')).reshape(2, 3).T
        elif dim == 3:
            phys_cell_vertices = numpy.asarray(sp.symbols('x:4,y:4,z:4')).reshape(3, 4).T

        gem_cell_vertices = coordinate_mapping.physical_vertices()

        phys_quad_points = sp.symbols('x,y,z')[:dim]
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
    basis = ([sp.Array([b, 0]) for b in lagrange]
             + [sp.Array([0, b]) for b in lagrange])
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

    AT = sp.Matrix(A).T
    coefficients = AT.inv('LU')
    return coefficients * sp.Matrix(basis)


def numbering(dimension):
    if dimension == 2:
        pass
    elif dimension == 3:
        raise NotImplementedError()

    # Counter clockwise for vertices on reference triangle (barycentre
    # is appended)
    cell2vertex = {0: [0, 1, 3],
                   1: [0, 2, 3],
                   2: [1, 2, 3]}

    # refinement is:
    # bottom edge, left edge, diagonal edge.
    # vertex dofs have global numbering first, then edges
    # local numbering is FIAT entity_dof convention for P2
    cell2node = {0: [0, 1, 3, 6, 5, 4],
                 1: [0, 2, 3, 8, 5, 7],
                 2: [1, 2, 3, 8, 6, 9]}

    return cell2vertex, cell2node


def integrate(subcell_vertices, expr, X):
    from sp.geometry.point import Point
    from sp.geometry.polygon import Polygon
    from sp.integrals.intpoly import polytope_integrate, x, y, z

    expr = expr.subs(dict(zip(X, (x, y, z))))
    polygon = Polygon(*(Point(*vtx) for vtx in subcell_vertices))
    flipsign = sp.simplify(polytope_integrate(polygon, expr=1)) < 0

    val = sp.simplify(polytope_integrate(polygon, expr))
    if flipsign:
        val = val * -1
    return val.subs(dict(zip((x, y, z), X)))


def beta(xy, f):
    """
        xy: array of [nvertices, dim] representing the coordinates
        f : index of face to calculate beta for,
            the face f is opposite vertex f
    """

    dim = 2
    fiat_cell = ufc_simplex(dim)
    topology = fiat_cell.get_topology()
    X = sp.symbols(f'x:{dim}')
    # FIXME: figure out monomial basis for P2 in arbitrary dimensions
    scalar_basis = [1 + 0*X[0], X[0], X[1], X[0]**2, X[0]*X[1], X[1]**2]

    # We need to make a basis for CG2 on the barycentric
    # refinement of our triangle.

    bary = xy.sum(axis=0)/(dim+1)
    meshxy = np.vstack([xy, bary])
    cell2vertex, cell2node = numbering(dim)
    CG2_basis = []
    for cell in range(dim+1):
        assert cell2vertex[cell] == cell2node[cell][:dim+1]
        dof_vertices = [methodcaller('subs', dict(zip(X, meshxy[cell2vertex[cell][i], :])))
                        for i in range(dim+1)]

        dof_edges = []
        for edge in range(len(list(combinations(range(dim+1), 2)))):
            # FIXME, use factorial for this
            v1, v2 = topology[1][edge]
            gv1 = cell2vertex[cell][v1]
            gv2 = cell2vertex[cell][v2]
            coord = 0.5*(meshxy[gv1, :] + meshxy[gv2, :])
            print(coord)
            dof = methodcaller('subs', dict(zip(X, coord)))
            dof_edges.append(dof)
        print("-"*80)

        dofs = dof_vertices + dof_edges

        A = [[dofi(basisj) for basisj in scalar_basis]
             for dofi in dofs]

        AT = sp.Matrix(A).T
        coefficients = sp.simplify(AT.inv('LU'))
        CG2_basis.append(coefficients * sp.Matrix(scalar_basis))

    m = np.zeros((20, 20), dtype=object)
    counter = count()
    # Equations on boundary (lagrange dofs)
    for node in [0, 1, 2, 4, 7, 9]:
        for i in range(dim):
            c = next(counter)
            m[c, dim*node + i] = 1 + 0*X[0]


    # Want div(B) = constant on each subcell
    # Obtained by orthogonality against the legendre polynomial of
    # degree 1. On [-1, 1], this is just X. On [0, 1] it's

    # i.e. \int div(B)*x = 0 and \int div(B) * y on each subcell
    V = [1 + X[0]*0] + [X[i] for i in range(dim)]

    def proj(u, v, xy):
        return u*integrate(xy, u*v, X) / integrate(xy, u*u, X)

    for cell in range(dim+1):
        # Do partial Gram-Schmidt to make an L2-orthogonal basis for
        # P1 on subcell.
        U = [V[0]]
        cellxy = meshxy[cell2vertex[cell], ...]
        # Just need to orthogonalise against 1, because we know X, Y,
        # and Z are already independent
        for s in range(dim):
            U.append(V[s+1] - proj(U[0], V[s+1], cellxy))

        # We now want to integrate div(B) against these polynomials to
        # get our constraint equations.
        # On this cell.
        # one row. column is int(div(basis_function)*U)
        # RHS is 0
    return CG2_basis, X


xy = np.array([[0, 0],
               [1, 0],
               [0, 1]])
# basis, X = beta(xy, 0)
