import FIAT
import numpy
from gem import ListTensor, Literal

from finat.fiat_elements import FiatElement
from finat.physically_mapped import Citations, PhysicallyMappedElement, adjugate
from copy import deepcopy


class BernardiRaugel(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree=None):
        sd = cell.get_spatial_dimension()
        if degree is None:
            degree = sd
        if degree != sd:
            raise ValueError(f"Bernardi-Raugel only defined for degree = {sd}")
        if Citations is not None:
            Citations().register("ArnoldQin1992")
        super().__init__(FIAT.BernardiRaugel(cell, degree))

        reduced_dofs = deepcopy(self._element.entity_dofs())
        for entity in reduced_dofs[sd-1]:
            reduced_dofs[sd-1][entity] = reduced_dofs[sd-1][entity][:1]
        self._entity_dofs = reduced_dofs

    def basis_transformation(self, coordinate_mapping):
        # Jacobians at cell center
        sd = self.cell.get_spatial_dimension()
        bary, = self.cell.make_points(sd, 0, sd+1)
        J = coordinate_mapping.jacobian_at(bary)
        adjJ = adjugate([[J[i, j] for j in range(sd)] for i in range(sd)])

        numbf = self._element.space_dimension()
        ndof = self.space_dimension()
        # rectangular to toss out the constraint dofs
        V = numpy.eye(numbf, ndof, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        edofs = self.entity_dofs()
        for v in sorted(edofs[0]):
            vdofs = edofs[0][v]
            V[numpy.ix_(vdofs, vdofs)] = adjJ

        fiat_cell = self.cell
        toffset = len(edofs[sd-1])

        if sd == 2:
            R = numpy.array([[0, 1], [-1, 0]])
            for f in sorted(edofs[sd-1]):
                that = fiat_cell.compute_edge_tangent(f)
                that /= numpy.linalg.norm(that)
                nhat = R @ that
                # Compute alpha and beta for the facet.
                Jn = J @ Literal(nhat)
                Jt = J @ Literal(that)
                alpha = Jn @ Jt
                beta = Jt @ Jt
                # Stuff the inverse into the right rows and columns.
                sn = edofs[sd-1][f][0]
                st = sn + toffset
                V[st, sn] = -1 * alpha / beta

        elif sd == 3:
            for f in sorted(edofs[sd-1]):
                # Compute the reciprocal basis
                thats = fiat_cell.compute_tangents(sd-1, f)
                nhat = numpy.cross(*thats)
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
                sn = edofs[sd-1][f][0]
                st1 = sn + toffset
                st2 = st1 + toffset
                V[st1, sn] = -1 * det1 / det0
                V[st2, sn] = -1 * det2 / det0

        return ListTensor(V.T)

    def entity_dofs(self):
        return self._entity_dofs

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    def space_dimension(self):
        return (self.cell.get_spatial_dimension() + 1)**2
