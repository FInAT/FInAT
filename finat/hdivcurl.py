from __future__ import absolute_import, division, print_function

from six import iteritems

from FIAT.reference_element import LINE

import gem
from finat.finiteelementbase import FiniteElementBase
from finat.tensor_product import TensorProductElement


class HDivElement(FiniteElementBase):
    def __init__(self, wrapped):
        assert isinstance(wrapped, TensorProductElement)
        if any(fe.formdegree is None for fe in wrapped.factors):
            raise ValueError("Form degree of subelement is None, cannot H(div)!")

        formdegree = sum(fe.formdegree for fe in wrapped.factors)
        if formdegree != wrapped.cell.get_spatial_dimension() - 1:
            raise ValueError("H(div) requires (n-1)-form element!")

        super(HDivElement, self).__init__()
        self.wrapped = wrapped
        self.transform = select_hdiv_transformer(wrapped)

    @property
    def cell(self):
        return self.wrapped.cell

    @property
    def degree(self):
        return self.wrapped.degree

    @property
    def formdegree(self):
        return self.cell.get_spatial_dimension() - 1

    def entity_dofs(self):
        return self.wrapped.entity_dofs()

    def entity_closure_dofs(self):
        return self.wrapped.entity_closure_dofs()

    def space_dimension(self):
        return self.wrapped.space_dimension()

    @property
    def index_shape(self):
        return self.wrapped.index_shape

    @property
    def value_shape(self):
        return (self.cell.get_spatial_dimension(),)

    def _transform_evaluation(self, core_eval):
        beta = self.get_indices()
        zeta = self.get_value_indices()

        def promote(table):
            v = gem.partial_indexed(table, beta)
            u = gem.ListTensor(self.transform(v))
            return gem.ComponentTensor(gem.Indexed(u, zeta), beta + zeta)

        return {alpha: promote(table)
                for alpha, table in iteritems(core_eval)}

    def basis_evaluation(self, order, ps, entity=None):
        core_eval = self.wrapped.basis_evaluation(order, ps, entity)
        return self._transform_evaluation(core_eval)

    def point_evaluation(self, order, refcoords, entity=None):
        core_eval = self.wrapped.point_evaluation(order, refcoords, entity)
        return self._transform_evaluation(core_eval)

    @property
    def mapping(self):
        return "contravariant piola"


def select_hdiv_transformer(element):
    # Assume: something x interval
    assert len(element.factors) == 2
    assert element.factors[1].cell.get_shape() == LINE

    ks = tuple(fe.formdegree for fe in element.factors)
    if ks == (0, 1):
        return lambda v: [gem.Product(gem.Literal(-1), v), gem.Zero()]
    elif ks == (1, 0):
        return lambda v: [gem.Zero(), v]
    elif ks == (2, 0):
        return lambda v: [gem.Zero(), gem.Zero(), v]
    elif ks == (1, 1):
        if element.mapping == "contravariant piola":
            return lambda v: [gem.Indexed(v, (0,)),
                              gem.Indexed(v, (1,)),
                              gem.Zero()]
        elif element.mapping == "covariant piola":
            return lambda v: [gem.Indexed(v, (1,)),
                              gem.Product(gem.Literal(-1), gem.Indexed(v, (0,))),
                              gem.Zero()]
        else:
            assert False, "Unexpected original mapping!"
    else:
        assert False, "Unexpected form degree combination!"
