"""Implementation of the Arnold-Winther finite elements."""
import FIAT
import numpy
from gem import ListTensor, Literal, partial_indexed

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

    bary = [1/(sd+1)] * sd
    detJ = coordinate_mapping.detJ_at(bary)
    J = coordinate_mapping.jacobian_at(bary)
    rns = coordinate_mapping.reference_normals()
    offset = dofs_per_facet
    if sd == 2:
        R = Literal(numpy.array([[0, -1], [1, 0]]))

        for e in range(num_facets):
            nhat = partial_indexed(rns, (e, ))
            that = R @ nhat
            Jn = J @ nhat
            Jt = J @ that

            # Compute alpha and beta for the edge.
            alpha = (Jn @ Jt) / detJ
            beta = (Jt @ Jt) / detJ
            # Stuff into the right rows and columns.
            for i in range(dimPk_facet):
                idx = offset*e + i * dimPk_facet + 1
                Vsub[idx, idx-1] = Literal(-1) * alpha / beta
                Vsub[idx, idx] = Literal(1) / beta
    elif sd == 3:
        pass
        for f in range(num_facets):
            nhat = fiat_cell.compute_normal(f)
            nhat /= numpy.linalg.norm(nhat)
            ehats = fiat_cell.compute_tangents(2, f)
            rels = [numpy.linalg.norm(ehat) for ehat in ehats]
            thats = [a / b for a, b in zip(ehats, rels)]
            vf = fiat_cell.volume_of_subcomplex(2, f)

            orth_vecs = [numpy.cross(nhat, thats[1]),
                         numpy.cross(thats[0], nhat)]

            Jn = J @ Literal(nhat)
            Jts = [J @ Literal(that) for that in thats]
            Jorth = [J @ Literal(ov) for ov in orth_vecs]

            alphas = [Literal(rels[i]) * (Jn @ Jts[i]) / detJ / Literal(vf) / Literal(2) for i in (0, 1)]
            betas = [Jorth[0] @ Jts[i] / detJ / Literal(thats[0] @ orth_vecs[0]) for i in (0, 1)]
            gammas = [Jorth[1] @ Jts[i] / detJ / Literal(thats[1] @ orth_vecs[1]) for i in (0, 1)]

            det = betas[0] * gammas[1] - betas[1] * gammas[0]

            for i in range(dimPk_facet):
                idx = offset*f + i * sd

                Vsub[idx+1, idx] = (alphas[1] * gammas[0]
                                    - alphas[0] * gammas[1]) / det
                Vsub[idx+1, idx+1] = gammas[1] / det
                Vsub[idx+1, idx+2] = Literal(-1) * gammas[0] / det
                Vsub[idx+2, idx] = (alphas[0] * betas[1]
                                    - alphas[1] * betas[0]) / det
                Vsub[idx+2, idx+1] = Literal(-1) * betas[1] / det
                Vsub[idx+2, idx+2] = betas[0] / det

    return Vsub


def _evaluation_transform(coordinate_mapping):
    J = coordinate_mapping.jacobian_at([1/3, 1/3])

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
    def __init__(self, cell, degree):
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
        W = _evaluation_transform(coordinate_mapping)
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
    def __init__(self, cell, degree):
        if Citations is not None:
            Citations().register("Arnold2002")
        super(ArnoldWinther, self).__init__(FIAT.ArnoldWinther(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        """The extra 6 dofs removed here correspond to the constraints."""
        V = numpy.zeros((30, 24), dtype=object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        W = _evaluation_transform(coordinate_mapping)

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
