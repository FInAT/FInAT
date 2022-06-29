from FIAT.hdivcurl import Hdiv, Hcurl
from FIAT.reference_element import LINE

import gem
from gem.utils import cached_property
from finat.finiteelementbase import FiniteElementBase
from finat.tensor_product import TensorProductElement


class WrapperElementBase(FiniteElementBase):
    """Common base class for H(div) and H(curl) element wrappers."""

    def __init__(self, wrappee, transform):
        super(WrapperElementBase, self).__init__()
        self.wrappee = wrappee
        """An appropriate tensor product FInAT element whose basis
        functions are mapped to produce an H(div) or H(curl)
        conforming element."""

        self.transform = transform
        """A transformation applied on the scalar/vector values of the
        wrapped element to produce an H(div) or H(curl) conforming
        element."""

    @property
    def cell(self):
        return self.wrappee.cell

    @property
    def degree(self):
        return self.wrappee.degree

    def entity_dofs(self):
        return self.wrappee.entity_dofs()

    @property
    def entity_permutations(self):
        return self.wrappee.entity_permutations

    def entity_closure_dofs(self):
        return self.wrappee.entity_closure_dofs()

    def entity_support_dofs(self):
        return self.wrappee.entity_support_dofs()

    def space_dimension(self):
        return self.wrappee.space_dimension()

    @property
    def index_shape(self):
        return self.wrappee.index_shape

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
                for alpha, table in core_eval.items()}

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        core_eval = self.wrappee.basis_evaluation(order, ps, entity)
        return self._transform_evaluation(core_eval)

    def point_evaluation(self, order, refcoords, entity=None):
        core_eval = self.wrappee.point_evaluation(order, refcoords, entity)
        return self._transform_evaluation(core_eval)

    @property
    def dual_basis(self):
        Q, x = self.wrappee.dual_basis
        beta = self.get_indices()
        zeta = self.get_value_indices()
        # Index out the basis indices from wrapee's Q, to get
        # something of wrappee.value_shape, then promote to new shape
        # with the same transform as done for basis evaluation
        Q = gem.ListTensor(self.transform(gem.partial_indexed(Q, beta)))
        # Finally wrap up Q in shape again (now with some extra
        # value_shape indices)
        return gem.ComponentTensor(Q[zeta], beta + zeta), x


class HDivElement(WrapperElementBase):
    """H(div) wrapper element for tensor product elements."""

    def __init__(self, wrappee):
        assert isinstance(wrappee, TensorProductElement)
        if any(fe.formdegree is None for fe in wrappee.factors):
            raise ValueError("Form degree of subelement is None, cannot H(div)!")

        formdegree = sum(fe.formdegree for fe in wrappee.factors)
        if formdegree != wrappee.cell.get_spatial_dimension() - 1:
            raise ValueError("H(div) requires (n-1)-form element!")

        transform = select_hdiv_transformer(wrappee)
        super(HDivElement, self).__init__(wrappee, transform)

    @property
    def formdegree(self):
        return self.cell.get_spatial_dimension() - 1

    @cached_property
    def fiat_equivalent(self):
        return Hdiv(self.wrappee.fiat_equivalent)

    @property
    def mapping(self):
        return "contravariant piola"


class HCurlElement(WrapperElementBase):
    """H(curl) wrapper element for tensor product elements."""

    def __init__(self, wrappee):
        assert isinstance(wrappee, TensorProductElement)
        if any(fe.formdegree is None for fe in wrappee.factors):
            raise ValueError("Form degree of subelement is None, cannot H(curl)!")

        formdegree = sum(fe.formdegree for fe in wrappee.factors)
        if formdegree != 1:
            raise ValueError("H(curl) requires 1-form element!")

        transform = select_hcurl_transformer(wrappee)
        super(HCurlElement, self).__init__(wrappee, transform)

    @property
    def formdegree(self):
        return 1

    @cached_property
    def fiat_equivalent(self):
        return Hcurl(self.wrappee.fiat_equivalent)

    @property
    def mapping(self):
        return "covariant piola"


def select_hdiv_transformer(element):
    # Assume: something x interval
    assert len(element.factors) == 2
    assert element.factors[1].cell.get_shape() == LINE

    # Globally consistent edge orientations of the reference
    # quadrilateral: rightward horizontally, upward vertically.
    # Their rotation by 90 degrees anticlockwise is interpreted as the
    # positive direction for normal vectors.
    ks = tuple(fe.formdegree for fe in element.factors)
    if ks == (0, 1):
        # Make the scalar value the leftward-pointing normal on the
        # y-aligned edges.
        return lambda v: [gem.Product(gem.Literal(-1), v), gem.Zero()]
    elif ks == (1, 0):
        # Make the scalar value the upward-pointing normal on the
        # x-aligned edges.
        return lambda v: [gem.Zero(), v]
    elif ks == (2, 0):
        # Same for 3D, so z-plane.
        return lambda v: [gem.Zero(), gem.Zero(), v]
    elif ks == (1, 1):
        if element.mapping == "contravariant piola":
            # Pad the 2-vector normal on the "base" cell into a
            # 3-vector, maintaining direction.
            return lambda v: [gem.Indexed(v, (0,)),
                              gem.Indexed(v, (1,)),
                              gem.Zero()]
        elif element.mapping == "covariant piola":
            # Rotate the 2-vector tangential component on the "base"
            # cell 90 degrees anticlockwise into a 3-vector and pad.
            return lambda v: [gem.Indexed(v, (1,)),
                              gem.Product(gem.Literal(-1), gem.Indexed(v, (0,))),
                              gem.Zero()]
        else:
            assert False, "Unexpected original mapping!"
    else:
        assert False, "Unexpected form degree combination!"


def select_hcurl_transformer(element):
    # Assume: something x interval
    assert len(element.factors) == 2
    assert element.factors[1].cell.get_shape() == LINE

    # Globally consistent edge orientations of the reference
    # quadrilateral: rightward horizontally, upward vertically.
    # Tangential vectors interpret these as the positive direction.
    dim = element.cell.get_spatial_dimension()
    ks = tuple(fe.formdegree for fe in element.factors)
    if element.mapping == "affine":
        if ks == (1, 0):
            # Can only be 2D.  Make the scalar value the
            # rightward-pointing tangential on the x-aligned edges.
            return lambda v: [v, gem.Zero()]
        elif ks == (0, 1):
            # Can be any spatial dimension.  Make the scalar value the
            # upward-pointing tangential.
            return lambda v: [gem.Zero()] * (dim - 1) + [v]
        else:
            assert False
    elif element.mapping == "covariant piola":
        # Second factor must be continuous interval.  Just padding.
        return lambda v: [gem.Indexed(v, (0,)),
                          gem.Indexed(v, (1,)),
                          gem.Zero()]
    elif element.mapping == "contravariant piola":
        # Second factor must be continuous interval.  Rotate the
        # 2-vector tangential component on the "base" cell 90 degrees
        # clockwise into a 3-vector and pad.
        return lambda v: [gem.Product(gem.Literal(-1), gem.Indexed(v, (1,))),
                          gem.Indexed(v, (0,)),
                          gem.Zero()]
    else:
        assert False, "Unexpected original mapping!"
