"""Implementation of the Arnold-Winther finite elements."""
import FIAT
import numpy
from gem import ListTensor

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, identity, PhysicallyMappedElement
from finat.piola_mapped import adjugate, normal_tangential_edge_transform, normal_tangential_face_transform


def _facet_transform(fiat_cell, facet_moment_degree, coordinate_mapping):
    sd = fiat_cell.get_spatial_dimension()
    top = fiat_cell.get_topology()
    num_facets = len(top[sd-1])
    dimPk_facet = FIAT.expansions.polynomial_dimension(
        fiat_cell.construct_subelement(sd-1), facet_moment_degree)
    dofs_per_facet = sd * dimPk_facet
    ndofs = num_facets * dofs_per_facet
    V = identity(ndofs)

    bary, = fiat_cell.make_points(sd, 0, sd+1)
    J = coordinate_mapping.jacobian_at(bary)
    detJ = coordinate_mapping.detJ_at(bary)
    if sd == 2:
        transform = normal_tangential_edge_transform
    elif sd == 3:
        transform = normal_tangential_face_transform

    for f in range(num_facets):
        rows = transform(fiat_cell, J, detJ, f)
        for i in range(dimPk_facet):
            s = dofs_per_facet*f + i * sd
            V[s+1:s+sd, s:s+sd] = rows
    return V


def _evaluation_transform(fiat_cell, coordinate_mapping):
    sd = fiat_cell.get_spatial_dimension()
    bary, = fiat_cell.make_points(sd, 0, sd+1)
    J = coordinate_mapping.jacobian_at(bary)
    K = adjugate([[J[i, j] for j in range(sd)] for i in range(sd)])

    indices = [(i, j) for i in range(sd) for j in range(i, sd)]
    ncomp = len(indices)
    W = numpy.zeros((ncomp, ncomp), dtype=object)
    for p, (i, j) in enumerate(indices):
        for q, (m, n) in enumerate(indices):
            W[p, q] = 0.5*(K[i, m] * K[j, n] + K[j, m] * K[i, n])
    W[:, [i != j for i, j in indices]] *= 2
    return W


class ArnoldWintherNC(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=2):
        if Citations is not None:
            Citations().register("Arnold2003")
        super().__init__(FIAT.ArnoldWintherNC(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """Note, the extra 3 dofs which are removed here
        correspond to the constraints."""
        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        V = identity(numbf, ndof)

        V[:12, :12] = _facet_transform(self.cell, 1, coordinate_mapping)

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

    def space_dimension(self):
        return 15


class ArnoldWinther(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=3):
        if Citations is not None:
            Citations().register("Arnold2002")
        super().__init__(FIAT.ArnoldWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        # The extra 6 dofs removed here correspond to the constraints
        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        V = identity(numbf, ndof)

        sd = self.cell.get_spatial_dimension()
        W = _evaluation_transform(self.cell, coordinate_mapping)
        ncomp = W.shape[0]

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W
        num_verts = sd + 1
        cur = num_verts * ncomp

        Vsub = _facet_transform(self.cell, 1, coordinate_mapping)
        fdofs = Vsub.shape[0]
        V[cur:cur+fdofs, cur:cur+fdofs] = Vsub
        cur += fdofs

        # RESCALING FOR CONDITIONING
        h = coordinate_mapping.cell_size()
        for e in range(num_verts):
            V[:, ncomp*e:ncomp*(e+1)] *= 1/(h[e] * h[e])

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

    def space_dimension(self):
        return 24
