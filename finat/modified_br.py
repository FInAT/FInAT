from itertools import combinations, count
from operator import methodcaller

import gem
import numpy
import numpy as np
import sympy
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
    from sympy.geometry.point import Point
    from sympy.geometry.polygon import Polygon
    from sympy.integrals.intpoly import polytope_integrate, x, y, z

    expr = expr.subs(dict(zip(X, (x, y, z))))
    polygon = Polygon(*(Point(*vtx) for vtx in subcell_vertices))
    flipsign = sympy.simplify(polytope_integrate(polygon, expr=1)) < 0

    val = sympy.simplify(polytope_integrate(polygon, expr))
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
    X = sympy.symbols(f'x:{dim}')
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

        AT = sympy.Matrix(A).T
        coefficients = sympy.simplify(AT.inv('LU'))
        CG2_basis.append(coefficients * sympy.Matrix(scalar_basis))

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

def beta_GM(xy, f):
    """
        xy: array of [nvertices, dim] representing the coordinates
        f : index of face to calculate beta for,
            the face f is opposite vertex f
    """

    assert f in [1,2,3]
    dim = 2
    fiat_cell = ufc_simplex(dim)
    topology = fiat_cell.get_topology()
    X = sympy.symbols(f'x:{dim}')

    n1 = 1/sympy.sqrt(2)*sympy.Matrix([1+0*X[0], 1+0*X[0]])   ########### Fix it
    n2 = -1*sympy.Matrix([1+0*X[0], 0*X[0]])
    n3 = -1*sympy.Matrix([0*X[0], 1+0*X[0]])
    A = sympy.Matrix([[1+0*X[0], 0*X[0]],[0*X[0], 1+0*X[0]]])   ########### Fix it
    s1 = A.inv()*n1
    s2 = A.inv()*n2
    s3 = A.inv()*n3

    w_K1, w_K2, w_K3 = get_w_K(f, X, s1, s2, s3)

    restrict_K2 = sympy.Piecewise(
                (sympy.Piecewise((1, X[1] < -2*X[0] + 1), (0,True)),  X[0] < X[1]),
                (0, True)
        )

    restrict_K3 = sympy.Piecewise(
                (sympy.Piecewise((1, X[1] < -0.5*X[0]+0.5+1.0e-10), (0,True)),  X[0] > X[1]-1.0e-10),
                (0, True)
        )

    restrict_K1 = sympy.Piecewise(
                (sympy.Piecewise((1,  X[1] > -2*X[0] + 1), (0,True)),   X[1] > -0.5*X[0]+0.5),
                (0, True)
        )

    w = restrict_K1 * w_K1 + restrict_K2 * w_K2 + restrict_K3 * w_K3

    lambda1 = 1 - X[0] - X[1]
    lambda2 = X[0]
    lambda3 = X[1]

    #This numbering is different to Guzman Neilan
    if f == 1:
        B = lambda2*lambda3
        n = n1
    elif f == 2:
        B = lambda3*lambda1
        n = n2
    elif f == 3:
        B = lambda1*lambda2
        n = n3

    beta = B*n - A*w

    return beta, X, w


def get_w_K(f, X, s1, s2, s3):
    lambda0_K2 = lambda phi : 3*phi[0]
    lambda0_K3 = lambda phi : 3*phi[1]
    lambda0_K1 = lambda phi : 3 - 3*phi[0] - 3*phi[1]

    #This numbering is different to Guzman Neilan
    if f == 1:
        s = s1
        w_K2 = -1/18 * lambda0_K2([X[0], X[1]]) * sympy.Matrix(
            [-s[0]*(3*X[0]+6*X[1]-2) + s[1]*2,
             -s[0]*(6*X[0]-6*X[1]-2) - s[1]*(3*X[0]+6*X[1]-2)]
            )
        w_K3 = -1/18 * lambda0_K3([X[0], X[1]]) * sympy.Matrix(
             [-s[0]*(6*X[0]+3*X[1]-2) - s[1]*(-6*X[0]+6*X[1]-2),
              s[0]*2 - s[1]*(6*X[0]+3*X[1]-2)]
            )
        w_K1 = -1/18 * lambda0_K1([X[0], X[1]]) * sympy.Matrix(
             [-s[0]*(9*X[0]+9*X[1]-5) - s[1]*(-12*X[0]-6*X[1]+4),
              s[0]*(6*X[0]+12*X[1]-4) + s[1]*(-9*X[0]-9*X[1]+5)]
            )
    elif f == 2:
        s = s2
        w_K2 = -1/18 * lambda0_K2([X[0], X[1]]) * sympy.Matrix(
            [s[0]*(3*X[0]+6*X[1]-2) + s[1]*(6*X[0]+12*X[1]-6),
             s[0]*(6*X[0]-6*X[1]-2) + s[1]*(15*X[0]-6*X[1]-6)]
            )
        w_K3 = -1/18 * lambda0_K3([X[0], X[1]]) * sympy.Matrix(
             [s[0]*(6*X[0]+3*X[1]-2) + s[1]*(6*X[0]+12*X[1]-6),
              -s[0]*2 + s[1]*(6*X[0]+3*X[1]-6)]
            )
        w_K1 = -1/18 * lambda0_K1([X[0], X[1]]) * sympy.Matrix(
            [s[0]*(9*X[0]+9*X[1]-5) + s[1]*(6*X[0]+12*X[1]-6),
             -s[0]*(6*X[0]+12*X[1]-4) - s[1]*(3*X[0]+15*X[1]-3)]
            )
    elif f == 3:
        s = s3
        w_K2 = -1/18 * lambda0_K2([X[0], X[1]]) * sympy.Matrix(
            [s[0]*(3*X[0]+6*X[1]-6) - s[1]*2,
             s[0]*(12*X[0]+6*X[1]-6) + s[1]*(3*X[0]+6*X[1]-2)]
            )
        w_K3 = -1/18 * lambda0_K3([X[0], X[1]]) * sympy.Matrix(
            [-s[0]*(6*X[0]-15*X[1]+6) - s[1]*(6*X[0]-6*X[1]+2),
             s[0]*(12*X[0]+6*X[1]-6) + s[1]*(6*X[0]+3*X[1]-2)]
            )
        w_K1 = -1/18 * lambda0_K1([X[0], X[1]]) * sympy.Matrix(
            [-s[0]*(15*X[0]+3*X[1]-3) - s[1]*(12*X[0]+6*X[1]-4),
             s[0]*(12*X[0]+6*X[1]-6) + s[1]*(9*X[0]+9*X[1]-5)]
            )

    return w_K1, w_K2, w_K3

xy = np.array([[0, 0],
               [1, 0],
               [0, 1]])
# basis, X = beta(xy, 0)
#beta = beta_GM(xy, 1)

if __name__ == "__main__":
    from firedrake import *
    from sympy2ufl import *
    import alfi
    xy = np.array([[0, 0],
               [1, 0],
               [0, 1]])
    f = 1
    beta, X, w = beta_GM(xy, f)
    base = firedrake.UnitTriangleMesh()
    mh = alfi.BaryMeshHierarchy(base, 0)
    mesh = mh[-1]

    V = VectorFunctionSpace(mesh, "CG", 2)
    V1 = FunctionSpace(mesh, "CG", 2)

    def convert2ufl(phi):
        phi = sympy.Array(phi)
        phi = sympy2ufl(phi, bindings={X[0]: SpatialCoordinate(mesh)[0], X[1]: SpatialCoordinate(mesh)[1]})
        phi = as_vector([phi[0][0], phi[1][0]])
        return phi

    beta = convert2ufl(beta)
    w = convert2ufl(w)

    beta = interpolate(beta, V)
    beta.rename("beta")

    w = interpolate(w, V)
    w.rename("w")

    pvd = File("beta.pvd")

    #compute correct beta
    x, y = SpatialCoordinate(mesh)
    lam0 = 1 - x - y
    lam1 = x
    lam2 = y

    if f ==1:
         n = Constant((1/sqrt(2), 1/sqrt(2)))
         bcdata = lam1 * lam2 * n
    elif f ==2:
         n = Constant((-1, 0))
         bcdata = lam0 * lam2 * n
    elif f ==3:
         n = Constant((0, -1))
         bcdata = lam0 * lam1 * n

    u = Function(V)
    J = 0.5 * inner(grad(div(u)), grad(div(u)))*dx + 0.5 * inner(jump(div(u)), jump(div(u)))*dS

    F = derivative(J, u, TestFunction(V))
    bc = DirichletBC(V, bcdata, "on_boundary")

    sp = {"ksp_type": "preonly",
          "snes_monitor": None,
          "pc_type": "svd",
          "pc_svd_monitor": None,
          "pc_factor_mat_solver_type": "mumps"}
    u.rename("correct_beta")
    solve(F == 0, u, bc, solver_parameters=sp)

    pvd.write(beta, w,  u)
