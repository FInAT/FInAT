from __future__ import absolute_import, print_function, division
from six import iteritems

import numpy

import FIAT

import gem

from finat.fiat_elements import ScalarFiatElement


class Morley(ScalarFiatElement):
    def __init__(self, cell):
        super(Morley, self).__init__(FIAT.Morley(cell))

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        assert coordinate_mapping is not None

        # Jacobians at edge midpoints
        J = coordinate_mapping.jacobian_at([1/3, 1/3])

        rns = [coordinate_mapping.reference_normal(i) for i in range(3)]
        pns = [coordinate_mapping.physical_normal(i) for i in range(3)]

        rts = [coordinate_mapping.reference_tangent(i) for i in range(3)]
        pts = [coordinate_mapping.physical_tangent(i) for i in range(3)]

        pel = coordinate_mapping.physical_edge_lengths()

        # B11 = rns[i, 0]*(pns[i, 0]*J[0,0] + pts[i, 0]*J[1, 0]) + rts[i, 0]*(pns[i, 0]*J[0, 1] + pts[i, 0]*J[1,1])

        # B12 = rns[i, 0]*(pns[i, 1]*J[0,0] + pts[i, 1]*J[1, 0]) + rts[i, 1]*(pns[i, 1]*J[0, 1] + pts[i, 1]*J[1,1])

        B11 = [gem.Sum(gem.Product(gem.Indexed(rns[i], (0, )),
                                   gem.Sum(gem.Product(gem.Indexed(pns[i], (0, )),
                                                       gem.Indexed(J, (0, 0))),
                                           gem.Product(gem.Indexed(pts[i], (0, )),
                                                       gem.Indexed(J, (1, 0))))),
                       gem.Product(gem.Indexed(rts[i], (0, )),
                                   gem.Sum(gem.Product(gem.Indexed(pns[i], (0, )),
                                                       gem.Indexed(J, (0, 1))),
                                           gem.Product(gem.Indexed(pts[i], (0, )),
                                                       gem.Indexed(J, (1, 1))))))
               for i in range(3)]

        B12 = [gem.Sum(gem.Product(gem.Indexed(rns[i], (0, )),
                                   gem.Sum(gem.Product(gem.Indexed(pns[i], (1, )),
                                                       gem.Indexed(J, (0, 0))),
                                           gem.Product(gem.Indexed(pts[i], (1, )),
                                                       gem.Indexed(J, (1, 0))))),
                       gem.Product(gem.Indexed(rts[i], (0, )),
                                   gem.Sum(gem.Product(gem.Indexed(pns[i], (1, )),
                                                       gem.Indexed(J, (0, 1))),
                                           gem.Product(gem.Indexed(pts[i], (1, )),
                                                       gem.Indexed(J, (1, 1))))))
               for i in range(3)]

        V = numpy.eye(6, dtype=object)
        for multiindex in numpy.ndindex(V.shape):
            V[multiindex] = gem.Literal(V[multiindex])

        for i in range(3):
            V[i + 3, i + 3] = B11[i]

        V[3, 1] = gem.Division(gem.Product(gem.Literal(-1), B12[0]), gem.Indexed(pel, (0, )))
        V[3, 2] = gem.Division(B12[0], gem.Indexed(pel, (0, )))

        V[4, 0] = gem.Division(gem.Product(gem.Literal(-1), B12[1]), gem.Indexed(pel, (1, )))
        V[4, 2] = gem.Division(B12[1], gem.Indexed(pel, (1, )))

        V[5, 0] = gem.Division(gem.Product(gem.Literal(-1), B12[2]), gem.Indexed(pel, (2, )))
        V[5, 1] = gem.Division(B12[2], gem.Indexed(pel, (2, )))

        M = V.T

        M = gem.ListTensor(M)

        def matvec(table):
            i = gem.Index()
            j = gem.Index()
            return gem.ComponentTensor(
                gem.IndexSum(gem.Product(gem.Indexed(M, (i, j)),
                                         gem.Indexed(table, (j,))),
                             (j,)),
                (i,))

        result = super(Morley, self).basis_evaluation(order, ps, entity=entity)
        return {alpha: matvec(table)
                for alpha, table in iteritems(result)}

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError  # TODO: think about it later!
