"""Implementation of the Hu-Zhang finite elements."""
import numpy
import FIAT
from gem import Literal, ListTensor
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement, Citations

def _edge_transform(T, coordinate_mapping, degree):
    p = degree # Just makes things easier to read
    #p = 3
    Vsub = numpy.zeros((6*(p - 1), 6*(p - 1)), dtype = object)

    for multiindex in numpy.ndindex(Vsub.shape):
        Vsub[multiindex] = Literal(Vsub[multiindex])

    for i in range(0, 6*(p - 1), 2):
        Vsub[i, i] = Literal(1)

    # This bypasses the GEM wrapper.
    that = numpy.array([T.compute_normalized_edge_tangent(i) for i in range(3)])
    nhat = numpy.array([T.compute_normal(i) for i in range(3)])

    detJ = coordinate_mapping.detJ_at([1/3, 1/3])
    J = coordinate_mapping.jacobian_at([1/3, 1/3])
    J_np = numpy.array([[J[0, 0], J[0, 1]],
                        [J[1, 0], J[1, 1]]])
    JTJ = J_np.T @ J_np

    for e in range(3):
        # Compute alpha and beta for the edge.
        Ghat_T = numpy.array([nhat[e, :], that[e, :]])

        (alpha, beta) = Ghat_T @ JTJ @ that[e, :] / detJ
        # Stuff into the right rows and columns.
        for deg in range(p - 1):
            (id1, id2) = (2*(p - 1)*e + 2*deg + 1, 2*(p - 1)*e + 2*deg + 3)
            Vsub[id1, id1 - 1] = Literal(-1) * alpha / beta
            Vsub[id1, id1] = Literal(1) / beta
            #Vsub[idx2, idx2 - 1] = Literal(-1) * alpha / beta
            #Vsub[idx2, idx2] = Literal(1) / beta

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

class HuZhang(PhysicallyMappedElement, FiatElement):
    def __init__(self, cell, degree):
        if Citations is not None:
            Citations().register("Hu2015")
        super(HuZhang, self).__init__(FIAT.HuZhang(cell, degree))

    def basis_transformation(self, coordinate_mapping):
        p = self.degree
        #p = 3
        #V = numpy.zeros((space_dimension(self), space_dimension(self)), dtype = object)
        #V = numpy.zeros((30, 30), dtype = object)
        dim = round(3*(p + 2)*(p + 1)/2)
        V = numpy.zeros((dim, dim), dtype = object)

        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = Literal(V[multiindex])

        W = _evaluation_transform(coordinate_mapping)

        # Put into the right rows and columns.
        V[0:3, 0:3] = V[3:6, 3:6] = V[6:9, 6:9] = W

        #V[9:21, 9:21] = _edge_transform(self.cell, coordinate_mapping, p)
        V[9:9 + 6*(p - 1), 9:9 + 6*(p - 1)] = _edge_transform(self.cell, coordinate_mapping, p)

        # internal DOFs
        detJ = coordinate_mapping.detJ_at([1/3, 1/3])
        #V[21:24, 21:24] = W / detJ
        ########## Can be done right later. Putting this as a temporary thing, which presumably gives better conditioning than having diagonal 1s in this block.
        #V[9 + 6*(p - 1):round(3*(p + 2)*(p + 1)/2), 9 + 6*(p - 1):round(3*(p + 2)*(p + 1)/2)] = W / detJ
        num_interior_dof_triples = round(p*(p - 1)/2) # NOTE divided by 3 since the DOFs come in 3s below
        for j in range(num_interior_dof_triples):
            #print(j)
            V[9 + 6*(p - 1) + 3*j:9 + 6*(p - 1) + 3*j + 3, 9 + 6*(p - 1) + 3*j:9 + 6*(p - 1) + 3*j + 3] = W / detJ
        #for j in range(3*num_interior_dof_triples):
        #    V[9 + 6*(p - 1) + j] = 1.0

#        # RESCALING FOR CONDITIONING
#        h = coordinate_mapping.cell_size()
#
#        for e in range(3):
#            eff_h = h[e]
#            for i in range(24):
#                V[i, 3*e] = V[i, 3*e]/(eff_h*eff_h)
#                V[i, 1 + 3*e] = V[i, 1 + 3*e]/(eff_h*eff_h)
#                V[i, 2 + 3*e] = V[i, 2 + 3*e]/(eff_h*eff_h)

        # Note: that the edge DOFs are scaled by edge lengths in FIAT implies
        # that they are already have the necessary rescaling to improve
        # conditioning.

        return ListTensor(V.T)

    def entity_dofs(self):
        p = self.degree
        return {0: {0: [0, 1, 2],
                    1: [3, 4, 5],
                    2: [6, 7, 8]},
                1: {0: [9 + j for j in range(2*(p - 1))],#[9, 10, 11, 12], 
                    1: [9 + 2*(p - 1) + j for j in range(2*(p - 1))],#[13, 14, 15, 16], 
                    2: [9 + 4*(p - 1) + j for j in range(2*(p - 1))]},#[17, 18, 19, 20]},
                2: {0: [9 + 6*(p - 1) + j for j in range(round(3*p*(p - 1)/2))]#[21, 22, 23]
                    }}

#    # need to overload since we're cutting out some dofs from the FIAT element.
#    def entity_closure_dofs(self):
#        ct = self.cell.topology
#        ecdofs = {i: {} for i in range(3)}
#        for i in range(3):
#            ecdofs[0][i] = list(range(3*i, 3*(i+1)))
#
#        for i in range(3):
#            ecdofs[1][i] = [dof for v in ct[1][i] for dof in ecdofs[0][v]] + \
#                list(range(9+4*i, 9+4*(i+1)))
#
#        ecdofs[2][0] = list(range(24))
#
#        return ecdofs

    @property
    def index_shape(self):
        p = self.degree
        return (round(3*(p + 2)*(p + 1)/2),)

    def space_dimension(self):
        p = self.degree
        return round(3*(p + 2)*(p + 1)/2)
