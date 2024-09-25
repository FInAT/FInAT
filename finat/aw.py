"""Implementation of the Arnold-Winther finite elements."""
import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement


def _facet_transform(fiat_cell, facet_moment_degree, coordinate_mapping):
    sd = fiat_cell.get_spatial_dimension()
    top = fiat_cell.get_topology()
    num_facets = len(top[sd-1])
    dimPk_facet = FIAT.expansions.polynomial_dimension(
        fiat_cell.construct_subelement(sd-1), facet_moment_degree)
    dofs_per_facet = sd * dimPk_facet
    ndofs = num_facets * dofs_per_facet

    Vsub = numpy.eye(ndofs, dtype=object)
    for multiindex in numpy.ndindex(Vsub.shape):
        Vsub[multiindex] = Literal(Vsub[multiindex])

    bary, = fiat_cell.make_points(sd, 0, sd+1)
    detJ = coordinate_mapping.detJ_at(bary)
    J = coordinate_mapping.jacobian_at(bary)
    if sd == 2:
        R = numpy.array([[0, 1], [-1, 0]])
        for f in range(num_facets):
            nhat = fiat_cell.compute_normal(f)
            that = R @ nhat
            # Compute alpha and beta for the facet.
            Jn = J @ Literal(nhat)
            Jt = J @ Literal(that)
            alpha = Jn @ Jt
            beta = Jt @ Jt
            # Stuff the inverse into the right rows and columns.
            row = (alpha / beta, detJ / beta)
            for i in range(dimPk_facet):
                s = dofs_per_facet*f + i * sd
                Vsub[s+1, s:s+sd] = row

    elif sd == 3:
        for f in range(num_facets):
            # Compute the reciprocal basis
            thats = fiat_cell.compute_tangents(sd-1, f)
            nhat = fiat_cell.compute_scaled_normal(f)
            nhat /= numpy.dot(nhat, nhat)
            orth_vecs = numpy.array([nhat,
                                     numpy.cross(nhat, thats[1]),
                                     numpy.cross(thats[0], nhat)])
            # Compute A = (alpha, beta, gamma) for the facet.
            Jts = J @ Literal(thats.T)
            Jorths = J @ Literal(orth_vecs.T)
            A = Jorths.T @ Jts
            # Stuff the inverse into the right rows and columns.
            det0 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
            det1 = A[2, 0] * A[0, 1] - A[2, 1] * A[0, 0]
            det2 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            scale = detJ / det0
            rows = ((det1 / det0, scale * A[2, 1], -1 * scale * A[2, 0]),
                    (det2 / det0, -1 * scale * A[1, 1], scale * A[1, 0]))
            for i in range(dimPk_facet):
                s = dofs_per_facet*f + i * sd
                Vsub[s+1:s+sd, s:s+sd] = rows
    return Vsub


def _evaluation_transform(fiat_cell, coordinate_mapping):
    sd = fiat_cell.get_spatial_dimension()
    bary, = fiat_cell.make_points(sd, 0, sd+1)
    J = coordinate_mapping.jacobian_at(bary)

    W = numpy.zeros((3, 3), dtype=object)
    W[0, 0] = J[1, 1]*J[1, 1]
    W[0, 1] = -2*J[1, 1]*J[0, 1]
    W[0, 2] = J[0, 1]*J[0, 1]
    W[1, 0] = -1*J[1, 1]*J[1, 0]
    W[1, 1] = J[1, 1]*J[0, 0] + J[0, 1]*J[1, 0]
    W[1, 2] = -1*J[0, 1]*J[0, 0]
    W[2, 0] = J[1, 0]*J[1, 0]
    W[2, 1] = -2*J[1, 0]*J[0, 0]
    W[2, 2] = J[0, 0]*J[0, 0]

    return W


class ArnoldWintherNC(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("Arnold2003")
        super(ArnoldWintherNC, self).__init__(FIAT.ArnoldWintherNC(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """Note, the extra 3 dofs which are removed here
        correspond to the constraints."""
        V = numpy.zeros((18, 15), dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        V[:12, :12] = _facet_transform(self.cell, 1, coordinate_mapping)

        # internal dofs
        W = _evaluation_transform(self.cell, coordinate_mapping)
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])

        V[12:15, 12:15] = W / detJ

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.

        return ListTensor(V.T)

    def entity_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]},
                2: {0: [12, 13, 14]}}

    def entity_closure_dofs(self):
        return {0: {0: [],
                    1: [],
                    2: []},
                1: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]},
                2: {0: list(range(15))}}

    @property
    def index_shape(self):
        return (15,)

    def space_dimension(self):
        return 15


class ArnoldWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3):
        if Citations is not None:
            Citations().register("Arnold2002")
        super(ArnoldWinther, self).__init__(FIAT.ArnoldWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """The extra 6 dofs removed here correspond to the constraints."""
        V = numpy.zeros((30, 24), dtype=object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        W = _evaluation_transform(self.cell, coordinate_mapping)

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W

        V[9:21, 9:21] = _facet_transform(self.cell, 1, coordinate_mapping)

        # internal DOFs
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])
        V[21:24, 21:24] = W / detJ

        # RESCALING FOR CONDITIONING
        h = coordinate_mapping.cell_size()

        for e in range(3):
            eff_h = h[e]
            for i in range(24):
                V[i, 3*e] = V[i, 3*e]/(eff_h*eff_h)
                V[i, 1+3*e] = V[i, 1+3*e]/(eff_h*eff_h)
                V[i, 2+3*e] = V[i, 2+3*e]/(eff_h*eff_h)

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.

        return ListTensor(V.T)

    def entity_dofs(self):
        return {0: {0: [0, 1, 2],
                    1: [3, 4, 5],
                    2: [6, 7, 8]},
                1: {0: [9, 10, 11, 12], 1: [13, 14, 15, 16], 2: [17, 18, 19, 20]},
                2: {0: [21, 22, 23]}}

    # need to overload since we're cutting out some dofs from the FIAT element.
    def entity_closure_dofs(self):
        ct = self.cell.topology
        ecdofs = {i: {} for i in range(3)}
        for i in range(3):
            ecdofs[0][i] = list(range(3*i, 3*(i+1)))

        for i in range(3):
            ecdofs[1][i] = [dof for v in ct[1][i] for dof in ecdofs[0][v]] + \
                list(range(9+4*i, 9+4*(i+1)))

        ecdofs[2][0] = list(range(24))

        return ecdofs

    @property
    def index_shape(self):
        return (24,)

    def space_dimension(self):
        return 24
