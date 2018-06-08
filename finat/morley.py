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

        # Jacobians at cell midpoints
        mps = [self.cell.make_points(1, i, 2)[-1] for i in range(3)]
        Js = [coordinate_mapping.jacobian_at(mp) for mp in mps]

        # how to get expr for length of local edge i?
        elens = [coordinate_mapping.edge_length(i) for i in range(3)]
        relens = [coordinate_mapping.ref_edge_tangent(i) for i in range(3)]
        
        d = 2
        numbf = 6
        
        M = numpy.eye(numbf, dtype=object)
        
        def n(J):
            assert J.shape == (d, d)
            return numpy.array(
                [[gem.Indexed(J, (i, j)) for j in range(d)]
                 for i in range(d)])

        for multiindex in numpy.ndindex(M.shape):
            M[multiindex] = gem.Literal(M[multiindex])

        B111
            
        # Now put the values into M!


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
