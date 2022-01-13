import numpy as np
import sympy as sp
from functools import singledispatch

import FIAT
from FIAT.polynomial_set import mis, form_matrix_product

import gem
from gem.utils import cached_property

from finat.finiteelementbase import FiniteElementBase
from finat.point_set import PointSet
from finat.sympy2gem import sympy2gem

try:
    from firedrake_citations import Citations
    Citations().add("Geevers2018new", """
@article{Geevers2018new,
 title={New higher-order mass-lumped tetrahedral elements for wave propagation modelling},
 author={Geevers, Sjoerd and Mulder, Wim A and van der Vegt, Jaap JW},
 journal={SIAM journal on scientific computing},
 volume={40},
 number={5},
 pages={A2830--A2857},
 year={2018},
 publisher={SIAM},
 doi={https://doi.org/10.1137/18M1175549},
}
""")
    Citations().add("Chin1999higher", """
@article{chin1999higher,
 title={Higher-order triangular and tetrahedral finite elements with mass lumping for solving the wave equation},
 author={Chin-Joe-Kong, MJS and Mulder, Wim A and Van Veldhuizen, M},
 journal={Journal of Engineering Mathematics},
 volume={35},
 number={4},
 pages={405--426},
 year={1999},
 publisher={Springer},
 doi={https://doi.org/10.1023/A:1004420829610},
}
""")
except ImportError:
    Citations = None


class FiatElement(FiniteElementBase):
    """Base class for finite elements for which the tabulation is provided
    by FIAT."""
    def __init__(self, fiat_element):
        super(FiatElement, self).__init__()
        self._element = fiat_element

    @property
    def cell(self):
        return self._element.get_reference_element()

    @property
    def degree(self):
        # Requires FIAT.CiarletElement
        return self._element.degree()

    @property
    def formdegree(self):
        return self._element.get_formdegree()

    def entity_dofs(self):
        return self._element.entity_dofs()

    def entity_dofs_per_derivative_order(self):
        return self._element.entity_dofs_per_derivative_order()

    def entity_closure_dofs(self):
        return self._element.entity_closure_dofs()

    @property
    def entity_permutations(self):
        return self._element.entity_permutations()

    def space_dimension(self):
        return self._element.space_dimension()

    @property
    def index_shape(self):
        return (self._element.space_dimension(),)

    @property
    def value_shape(self):
        return self._element.value_shape()

    @property
    def fiat_equivalent(self):
        # Just return the underlying FIAT element
        return self._element

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set.
        :param entity: the cell entity on which to tabulate.
        '''
        space_dimension = self._element.space_dimension()
        value_size = np.prod(self._element.value_shape(), dtype=int)
        fiat_result = self._element.tabulate(order, ps.points, entity)
        result = {}
        # In almost all cases, we have
        # self.space_dimension() == self._element.space_dimension()
        # But for Bell, FIAT reports 21 basis functions,
        # but FInAT only 18 (because there are actually 18
        # basis functions, and the additional 3 are for
        # dealing with transformations between physical
        # and reference space).
        index_shape = (self._element.space_dimension(),)
        for alpha, fiat_table in fiat_result.items():
            if isinstance(fiat_table, Exception):
                result[alpha] = gem.Failure(self.index_shape + self.value_shape, fiat_table)
                continue

            derivative = sum(alpha)
            table_roll = fiat_table.reshape(
                space_dimension, value_size, len(ps.points)
            ).transpose(1, 2, 0)

            exprs = []
            for table in table_roll:
                if derivative < self.degree:
                    point_indices = ps.indices
                    point_shape = tuple(index.extent for index in point_indices)

                    exprs.append(gem.partial_indexed(
                        gem.Literal(table.reshape(point_shape + index_shape)),
                        point_indices
                    ))
                elif derivative == self.degree:
                    # Make sure numerics satisfies theory
                    exprs.append(gem.Literal(table[0]))
                else:
                    # Make sure numerics satisfies theory
                    assert np.allclose(table, 0.0)
                    exprs.append(gem.Zero(self.index_shape))
            if self.value_shape:
                # As above, this extent may be different from that
                # advertised by the finat element.
                beta = tuple(gem.Index(extent=i) for i in index_shape)
                assert len(beta) == len(self.get_indices())

                zeta = self.get_value_indices()
                result[alpha] = gem.ComponentTensor(
                    gem.Indexed(
                        gem.ListTensor(np.array(
                            [gem.Indexed(expr, beta) for expr in exprs]
                        ).reshape(self.value_shape)),
                        zeta),
                    beta + zeta
                )
            else:
                expr, = exprs
                result[alpha] = expr
        return result

    def point_evaluation(self, order, refcoords, entity=None):
        '''Return code for evaluating the element at an arbitrary points on
        the reference element.

        :param order: return derivatives up to this order.
        :param refcoords: GEM expression representing the coordinates
                          on the reference entity.  Its shape must be
                          a vector with the correct dimension, its
                          free indices are arbitrary.
        :param entity: the cell entity on which to tabulate.
        '''
        if entity is None:
            entity = (self.cell.get_dimension(), 0)
        entity_dim, entity_i = entity

        # Spatial dimension of the entity
        esd = self.cell.construct_subelement(entity_dim).get_spatial_dimension()
        assert isinstance(refcoords, gem.Node) and refcoords.shape == (esd,)

        # Dispatch on FIAT element class
        return point_evaluation(self._element, order, refcoords, (entity_dim, entity_i))

    @cached_property
    def _dual_basis(self):
        # Return the numerical part of the dual basis, this split is
        # needed because the dual_basis itself can't produce the same
        # point set over and over in case it is used multiple times
        # (in for example a tensorproductelement).
        fiat_dual_basis = self._element.dual_basis()
        seen = dict()
        allpts = []
        # Find the unique points to evaluate at.
        # We might be able to make this a smaller set by treating each
        # point one by one, but most of the redundancy comes from
        # multiple functionals using the same quadrature rule.
        for dual in fiat_dual_basis:
            if len(dual.deriv_dict) != 0:
                raise NotImplementedError("FIAT dual bases with derivative nodes represented via a ``Functional.deriv_dict`` property do not currently have a FInAT dual basis")
            pts = dual.get_point_dict().keys()
            pts = tuple(sorted(pts))  # need this for determinism
            if pts not in seen:
                # k are indices into Q (see below) for the seen points
                kstart = len(allpts)
                kend = kstart + len(pts)
                seen[pts] = kstart, kend
                allpts.extend(pts)
        # Build Q.
        # Q is a tensor of weights (of total rank R) to contract with a unique
        # vector of points to evaluate at, giving a tensor (of total rank R-1)
        # where the first indices (rows) correspond to a basis functional
        # (node).
        # Q is a DOK Sparse matrix in (row, col, higher,..)=>value pairs (to
        # become a gem.SparseLiteral when implemented).
        # Rows (i) are number of nodes/dual functionals.
        # Columns (k) are unique points to evaluate.
        # Higher indices (*cmp) are tensor indices of the weights when weights
        # are tensor valued.
        Q = {}
        for i, dual in enumerate(fiat_dual_basis):
            point_dict = dual.get_point_dict()
            pts = tuple(sorted(point_dict.keys()))
            kstart, kend = seen[pts]
            for p, k in zip(pts, range(kstart, kend)):
                for weight, cmp in point_dict[p]:
                    Q[(i, k, *cmp)] = weight
        if all(len(set(key)) == 1 and np.isclose(weight, 1) and len(key) == 2
               for key, weight in Q.items()):
            # Identity matrix Q can be expressed symbolically
            extents = tuple(map(max, zip(*Q.keys())))
            js = tuple(gem.Index(extent=e+1) for e in extents)
            assert len(js) == 2
            Q = gem.ComponentTensor(gem.Delta(*js), js)
        else:
            # temporary until sparse literals are implemented in GEM which will
            # automatically convert a dictionary of keys internally.
            # TODO the below is unnecessarily slow and would be sped up
            # significantly by building Q in a COO format rather than DOK (i.e.
            # storing coords and associated data in (nonzeros, entries) shaped
            # numpy arrays) to take advantage of numpy multiindexing
            Qshape = tuple(s + 1 for s in map(max, *Q))
            Qdense = np.zeros(Qshape, dtype=np.float64)
            for idx, value in Q.items():
                Qdense[idx] = value
            Q = gem.Literal(Qdense)
        return Q, np.asarray(allpts)

    @property
    def dual_basis(self):
        # Return Q with x.indices already a free index for the
        # consumer to use
        # expensive numerical extraction is done once per element
        # instance, but the point set must be created every time we
        # build the dual.
        Q, pts = self._dual_basis
        x = PointSet(pts)
        assert len(x.indices) == 1
        assert Q.shape[1] == x.indices[0].extent
        i, *js = gem.indices(len(Q.shape) - 1)
        Q = gem.ComponentTensor(gem.Indexed(Q, (i, *x.indices, *js)), (i, *js))
        return Q, x

    @property
    def mapping(self):
        mappings = set(self._element.mapping())
        if len(mappings) != 1:
            return None
        else:
            result, = mappings
            return result


@singledispatch
def point_evaluation(fiat_element, order, refcoords, entity):
    raise AssertionError("FIAT element expected!")


@point_evaluation.register(FIAT.FiniteElement)
def point_evaluation_generic(fiat_element, order, refcoords, entity):
    # Coordinates on the reference entity (SymPy)
    esd, = refcoords.shape
    Xi = sp.symbols('X Y Z')[:esd]

    space_dimension = fiat_element.space_dimension()
    value_size = np.prod(fiat_element.value_shape(), dtype=int)
    fiat_result = fiat_element.tabulate(order, [Xi], entity)
    result = {}
    for alpha, fiat_table in fiat_result.items():
        if isinstance(fiat_table, Exception):
            result[alpha] = gem.Failure((space_dimension,) + fiat_element.value_shape(), fiat_table)
            continue

        # Convert SymPy expression to GEM
        mapper = gem.node.Memoizer(sympy2gem)
        mapper.bindings = {s: gem.Indexed(refcoords, (i,))
                           for i, s in enumerate(Xi)}
        gem_table = np.vectorize(mapper)(fiat_table)

        table_roll = gem_table.reshape(space_dimension, value_size).transpose()

        exprs = []
        for table in table_roll:
            exprs.append(gem.ListTensor(table.reshape(space_dimension)))
        if fiat_element.value_shape():
            beta = (gem.Index(extent=space_dimension),)
            zeta = tuple(gem.Index(extent=d)
                         for d in fiat_element.value_shape())
            result[alpha] = gem.ComponentTensor(
                gem.Indexed(
                    gem.ListTensor(np.array(
                        [gem.Indexed(expr, beta) for expr in exprs]
                    ).reshape(fiat_element.value_shape())),
                    zeta),
                beta + zeta
            )
        else:
            expr, = exprs
            result[alpha] = expr
    return result


@point_evaluation.register(FIAT.CiarletElement)
def point_evaluation_ciarlet(fiat_element, order, refcoords, entity):
    # Coordinates on the reference entity (SymPy)
    esd, = refcoords.shape
    Xi = sp.symbols('X Y Z')[:esd]

    # Coordinates on the reference cell
    cell = fiat_element.get_reference_element()
    X = cell.get_entity_transform(*entity)(Xi)

    # Evaluate expansion set at SymPy point
    poly_set = fiat_element.get_nodal_basis()
    degree = poly_set.get_embedded_degree()
    base_values = poly_set.get_expansion_set().tabulate(degree, [X])
    m = len(base_values)
    assert base_values.shape == (m, 1)
    base_values_sympy = np.array(list(base_values.flat))

    # Find constant polynomials
    def is_const(expr):
        try:
            float(expr)
            return True
        except TypeError:
            return False
    const_mask = np.array(list(map(is_const, base_values_sympy)))

    # Convert SymPy expression to GEM
    mapper = gem.node.Memoizer(sympy2gem)
    mapper.bindings = {s: gem.Indexed(refcoords, (i,))
                       for i, s in enumerate(Xi)}
    base_values = gem.ListTensor(list(map(mapper, base_values.flat)))

    # Populate result dict, creating precomputed coefficient
    # matrices for each derivative tuple.
    result = {}
    for i in range(order + 1):
        for alpha in mis(cell.get_spatial_dimension(), i):
            D = form_matrix_product(poly_set.get_dmats(), alpha)
            table = np.dot(poly_set.get_coeffs(), np.transpose(D))
            assert table.shape[-1] == m
            zerocols = np.isclose(abs(table).max(axis=tuple(range(table.ndim - 1))), 0.0)
            if all(np.logical_or(const_mask, zerocols)):
                # Casting is safe by assertion of is_const
                vals = base_values_sympy[const_mask].astype(np.float64)
                result[alpha] = gem.Literal(table[..., const_mask].dot(vals))
            else:
                beta = tuple(gem.Index() for s in table.shape[:-1])
                k = gem.Index()
                result[alpha] = gem.ComponentTensor(
                    gem.IndexSum(
                        gem.Product(gem.Indexed(gem.Literal(table), beta + (k,)),
                                    gem.Indexed(base_values, (k,))),
                        (k,)
                    ),
                    beta
                )
    return result


class Regge(FiatElement):  # naturally tensor valued
    def __init__(self, cell, degree):
        super(Regge, self).__init__(FIAT.Regge(cell, degree))


class HellanHerrmannJohnson(FiatElement):  # symmetric matrix valued
    def __init__(self, cell, degree):
        super(HellanHerrmannJohnson, self).__init__(FIAT.HellanHerrmannJohnson(cell, degree))


class ScalarFiatElement(FiatElement):
    @property
    def value_shape(self):
        return ()


class Bernstein(ScalarFiatElement):
    # TODO: Replace this with a smarter implementation
    def __init__(self, cell, degree):
        super().__init__(FIAT.Bernstein(cell, degree))


class Bubble(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(Bubble, self).__init__(FIAT.Bubble(cell, degree))


class FacetBubble(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(FacetBubble, self).__init__(FIAT.FacetBubble(cell, degree))


class CrouzeixRaviart(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(CrouzeixRaviart, self).__init__(FIAT.CrouzeixRaviart(cell, degree))


class Lagrange(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(Lagrange, self).__init__(FIAT.Lagrange(cell, degree))


class KongMulderVeldhuizen(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(KongMulderVeldhuizen, self).__init__(FIAT.KongMulderVeldhuizen(cell, degree))
        if Citations is not None:
            Citations().register("Chin1999higher")
            Citations().register("Geevers2018new")


class DiscontinuousLagrange(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(DiscontinuousLagrange, self).__init__(FIAT.DiscontinuousLagrange(cell, degree))


class Real(DiscontinuousLagrange):
    ...


class Serendipity(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(Serendipity, self).__init__(FIAT.Serendipity(cell, degree))


class DPC(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(DPC, self).__init__(FIAT.DPC(cell, degree))


class DiscontinuousTaylor(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(DiscontinuousTaylor, self).__init__(FIAT.DiscontinuousTaylor(cell, degree))


class VectorFiatElement(FiatElement):
    @property
    def value_shape(self):
        return (self.cell.get_spatial_dimension(),)


class RaviartThomas(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super(RaviartThomas, self).__init__(FIAT.RaviartThomas(cell, degree, variant=variant))


class BrezziDouglasMarini(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super(BrezziDouglasMarini, self).__init__(FIAT.BrezziDouglasMarini(cell, degree, variant=variant))


class BrezziDouglasFortinMarini(VectorFiatElement):
    def __init__(self, cell, degree):
        super(BrezziDouglasFortinMarini, self).__init__(FIAT.BrezziDouglasFortinMarini(cell, degree))


class Nedelec(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super(Nedelec, self).__init__(FIAT.Nedelec(cell, degree, variant=variant))


class NedelecSecondKind(VectorFiatElement):
    def __init__(self, cell, degree, variant=None):
        super(NedelecSecondKind, self).__init__(FIAT.NedelecSecondKind(cell, degree, variant=variant))
