from __future__ import absolute_import, print_function, division
from six import iteritems

import numpy

import FIAT

import gem

from finat.fiat_elements import ScalarFiatElement


class CubicHermite(ScalarFiatElement):
    def __init__(self, cell):
        super(CubicHermite, self).__init__(FIAT.CubicHermite(cell))

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        assert coordinate_mapping is not None
        JA, JB, JC = [coordinate_mapping.jacobian_at(vertex)
                      for vertex in self.cell.get_vertices()]

        def n(J):
            assert J.shape == (2, 2)
            return numpy.array(
                [[gem.Indexed(J, (0, 0)),
                  gem.Indexed(J, (0, 1))],
                 [gem.Indexed(J, (1, 0)),
                  gem.Indexed(J, (1, 1))]]
            )
        M = numpy.eye(10, dtype=object)
        for multiindex in numpy.ndindex(M.shape):
            M[multiindex] = gem.Literal(M[multiindex])
        M[1:3, 1:3] = n(JA)
        M[4:6, 4:6] = n(JB)
        M[7:9, 7:9] = n(JC)
        M = gem.ListTensor(M)

        def matvec(table):
            i = gem.Index()
            j = gem.Index()
            return gem.ComponentTensor(gem.IndexSum(gem.Product(gem.Indexed(M, (i, j)),
                                                                gem.Indexed(table, (j,))),
                                                    (j,)),
                                       (i,))

        result = super(CubicHermite, self).basis_evaluation(order, ps, entity=entity)
        return {alpha: matvec(table)
                for alpha, table in iteritems(result)}

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError  # TODO: think about it later!
