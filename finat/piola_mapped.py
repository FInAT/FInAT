import numpy
from gem import Literal


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
    """Return the basis transformation of evaluation at a point"""
    sd = fiat_cell.get_spatial_dimension()
    Jnp = numpy.array([[J[i, j] for j in range(sd)] for i in range(sd)])
    return adjugate(Jnp)


def normal_tangential_edge_transform(fiat_cell, J, detJ, f):
    """Return the basis transformation of
    normal and tangential edge moments"""
    R = numpy.array([[0, 1], [-1, 0]])
    that = fiat_cell.compute_edge_tangent(f)
    that /= numpy.linalg.norm(that)
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
