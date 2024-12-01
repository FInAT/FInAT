import numpy

from FIAT.functional import PointDivergence, FrobeniusIntegralMoment
from finat.fiat_elements import FiatElement
from finat.physically_mapped import identity, PhysicallyMappedElement
from gem import Literal, ListTensor, Power
from copy import deepcopy


def determinant(A):
    """Return the determinant of A"""
    n = A.shape[0]
    if n == 0:
        return 1
    elif n == 1:
        return A[0, 0]
    elif n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    else:
        detA = A[0, 0] * determinant(A[1:, 1:])
        cols = numpy.ones(A.shape[1], dtype=bool)
        for j in range(1, n):
            cols[j] = False
            detA += (-1)**j * A[0, j] * determinant(A[1:][:, cols])
            cols[j] = True
        return detA


def adjugate(A):
    """Return the adjugate matrix of A"""
    A = numpy.asarray(A)
    C = numpy.zeros_like(A)
    rows = numpy.ones(A.shape[0], dtype=bool)
    cols = numpy.ones(A.shape[1], dtype=bool)
    for i in range(A.shape[0]):
        rows[i] = False
        for j in range(A.shape[1]):
            cols[j] = False
            C[j, i] = (-1)**(i+j)*determinant(A[rows, :][:, cols])
            cols[j] = True
        rows[i] = True
    return C


def piola_inverse(fiat_cell, J, detJ):
    """Return the basis transformation of evaluation at a point.
    This simply inverts the Piola transform inv(J / detJ) = adj(J)."""
    sd = fiat_cell.get_spatial_dimension()
    Jnp = numpy.array([[J[i, j] for j in range(sd)] for i in range(sd)])
    return adjugate(Jnp)


def normal_tangential_edge_transform(fiat_cell, J, detJ, f):
    """Return the basis transformation of
    normal and tangential edge moments"""
    R = numpy.array([[0, 1], [-1, 0]])
    that = fiat_cell.compute_normalized_edge_tangent(f)
    nhat = R @ that
    Jn = J @ Literal(nhat)
    Jt = J @ Literal(that)
    alpha = Jn @ Jt
    beta = Jt @ Jt
    # Compute the last row of inv([[1, 0], [alpha/detJ, beta/detJ]])
    row = (-1 * alpha / beta, detJ / beta)
    return row


def normal_tangential_face_transform(fiat_cell, J, detJ, f):
    """Return the basis transformation of
    normal and tangential face moments"""
    # Compute the reciprocal basis
    thats = fiat_cell.compute_tangents(2, f)
    nhat = numpy.cross(*thats)
    nhat /= numpy.dot(nhat, nhat)
    orth_vecs = numpy.array([nhat,
                             numpy.cross(nhat, thats[1]),
                             numpy.cross(thats[0], nhat)])
    # Compute A = (alpha, beta, gamma)
    Jts = J @ Literal(thats.T)
    Jorths = J @ Literal(orth_vecs.T)
    A = Jorths.T @ Jts
    # Compute the last two rows of inv([[1, 0, 0], A.T/detJ])
    det0 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    det1 = A[2, 0] * A[0, 1] - A[2, 1] * A[0, 0]
    det2 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    scale = detJ / det0
    rows = ((-1 * det1 / det0, -1 * scale * A[2, 1], scale * A[2, 0]),
            (-1 * det2 / det0, scale * A[1, 1], -1 * scale * A[1, 0]))
    return rows


class PiolaMappedElement(PhysicallyMappedElement, FiatElement):
    """A general class to transform H1 Piola-mapped elements."""
    def __init__(self, fiat_element):
        mapping, = set(fiat_element.mapping())
        if mapping != "contravariant piola":
            raise ValueError(f"{type(fiat_element).__name__} needs to be Piola mapped.")
        super().__init__(fiat_element)

    def basis_transformation(self, coordinate_mapping):
        nodes = self._element.dual_basis()
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        dofs = self.entity_dofs()
        V = identity(self.space_dimension())
        one = V[0, 0]

        Finv = piola_inverse(self.cell, J, detJ)
        if sd == 2:
            transform = normal_tangential_edge_transform
        elif sd == 3:
            transform = normal_tangential_face_transform

        half = Literal(0.5)
        for dim in dofs:
            if dim == sd:
                continue

            for entity in sorted(dofs[dim]):
                edofs = dofs[dim][entity]
                if dim == sd-1:
                    thats = self.cell.compute_tangents(dim, entity)
                    nhat = numpy.dot([[0, 1], [-1, 0]], *thats) if sd == 2 else numpy.cross(*thats)
                    ref_norms = numpy.linalg.norm((nhat, *thats), axis=1)

                    Jt = J @ Literal(thats.T)
                    G = Jt.T @ Jt
                    phys_norms = [Power(determinant(G), half)]
                    phys_norms.extend(Power(G[i, i], half) for i in range(dim))
                    scale = [phys_norms[i] / ref_norms[i] for i in range(dim+1)]
                    scale = [one]*len(scale)

                    rows = identity(sd)
                    rows[1:] = transform(self.cell, J, detJ, entity)
                    for j in range(rows.shape[1]):
                        rows[:, j] *= scale[j]

                k = 0
                while k < len(edofs):
                    s = edofs[k:k+sd]
                    if dim == sd-1 and isinstance(nodes[s[0]], FrobeniusIntegralMoment):
                        # Deal with normal and tangential moments
                        V[numpy.ix_(s, s)] = rows
                    else:
                        # Undo the Piola transform
                        V[numpy.ix_(s, s)] = Finv
                    k += len(s)

        return ListTensor(V.T)


class PiolaBubbleElement(PhysicallyMappedElement, FiatElement):
    """A general class to transform H1 Piola-mapped elements with normal facet bubbles."""
    def __init__(self, fiat_element):
        mapping, = set(fiat_element.mapping())
        if mapping != "contravariant piola":
            raise ValueError(f"{type(fiat_element).__name__} needs to be Piola mapped.")
        super().__init__(fiat_element)

        # On each facet we expect the normal dof followed by the tangential ones
        # The tangential dofs should be numbered last, and are constrained to be zero
        sd = self.cell.get_spatial_dimension()
        reduced_dofs = deepcopy(self._element.entity_dofs())
        reduced_dim = 0
        cur = reduced_dofs[sd-1][0][0]
        for entity in sorted(reduced_dofs[sd-1]):
            reduced_dim += len(reduced_dofs[sd-1][entity][1:])
            reduced_dofs[sd-1][entity] = [cur]
            cur += 1
        self._entity_dofs = reduced_dofs
        self._space_dimension = fiat_element.space_dimension() - reduced_dim

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension

    def basis_transformation(self, coordinate_mapping):
        nodes = self._element.dual_basis()
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        detJ = coordinate_mapping.detJ_at(bary)

        dofs = self.entity_dofs()
        bfs = self._element.entity_dofs()
        ndof = self.space_dimension()
        numbf = self._element.space_dimension()
        V = identity(numbf, ndof)

        # Undo the Piola transform for non-facet bubble basis functions
        Finv = piola_inverse(self.cell, J, detJ)
        for dim in dofs:
            if dim == sd-1:
                continue
            for e in sorted(dofs[dim]):
                k = 0
                while k < len(dofs[dim][e]):
                    cur = dofs[dim][e][k]
                    if isinstance(nodes[cur], PointDivergence):
                        V[cur, cur] = detJ
                        k += 1
                    else:
                        s = dofs[dim][e][k:k+sd]
                        V[numpy.ix_(s, s)] = Finv
                        k += sd

        # Unpick the normal component for the facet bubbles
        if sd == 2:
            transform = normal_tangential_edge_transform
        elif sd == 3:
            transform = normal_tangential_face_transform

        for f in sorted(dofs[sd-1]):
            rows = numpy.asarray(transform(self.cell, J, detJ, f))
            k = 0
            num_facet_dofs = len(dofs[sd-1][f])
            num_facet_bfs = len(bfs[sd-1][f])
            while k < num_facet_bfs:
                cur_bfs = bfs[sd-1][f][k+1:k+sd]
                cur_dofs = dofs[sd-1][f][k:max(k+sd, num_facet_dofs)]
                V[numpy.ix_(cur_bfs, cur_dofs)] = rows[..., :len(cur_dofs)]
                k += sd
        return ListTensor(V.T)
