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
        Js = [coordinate_mapping.jacobian_at(vertex)
              for vertex in self.cell.get_vertices()]

        d = self.cell.get_dimension()
        numbf = self.space_dimension()

        def n(J):
            assert J.shape == (d, d)
            return numpy.array(
                [[gem.Indexed(J, (i, j)) for j in range(d)]
                 for i in range(d)])

        M = numpy.eye(numbf, dtype=object)

        for multiindex in numpy.ndindex(M.shape):
            M[multiindex] = gem.Literal(M[multiindex])

        cur = 0
        for i in range(d+1):
            cur += 1  # skip the vertex
            M[cur:cur+d, cur:cur+d] = n(Js[i])
            cur += d

        M = gem.ListTensor(M)

        def matvec(table):
            i = gem.Index()
            j = gem.Index()
            val = gem.ComponentTensor(
                gem.IndexSum(gem.Product(gem.Indexed(M, (i, j)),
                                         gem.Indexed(table, (j,))),
                             (j,)),
                (i,))
            # Eliminate zeros
            return gem.optimise.aggressive_unroll(val)

        result = super(CubicHermite, self).basis_evaluation(order, ps, entity=entity)
        return {alpha: matvec(table)
                for alpha, table in iteritems(result)}

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError  # TODO: think about it later!
