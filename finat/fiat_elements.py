import numpy as np
import sympy as sp
import collections
from functools import singledispatch

import FIAT
from FIAT.polynomial_set import mis, form_matrix_product

import gem

from finat.finiteelementbase import FiniteElementBase
from finat.sympy2gem import sympy2gem


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

    def entity_closure_dofs(self):
        return self._element.entity_closure_dofs()

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

    def dual_basis(self):
        '''Returns a tuple where each element of the tuple represents one
        functional in the dual shape. Each functional is represented by
        a tuple of tuple containing the points (PointSet), weights
        (Indexed Literal), multi_indices for extracting alpha from ReferenceGrad
        (tuple), and components (tuple), sorted by total derivative order.
        Can be thought of as a tuples of tuples of flattened tuples to avoid
        messy dictionary key-item pairings.

        For example, a dual basis containing 2 functionals with maximum derivative
        order of 1 would be represented by:
        (((point_set_10, weights_10, multi_indices_10, cmp_10),
          (point_set_11, weights_11, multi_indices_11, cmp_11))
         ((point_set_20, weights_20, multi_indices_20, cmp_20),
          ()))
        where one of the innermost tuples are empty because there are no evaluations
        correponding to that order.
        '''
        from itertools import chain
        from finat.point_set import PointSet

        max_deriv_order = max([ell.max_deriv_order for ell in self._element.dual_basis()])

        duals = []
        point_set_cache = {}    # To avoid repeating points?
        for dual in self._element.dual_basis():
            derivs = []
            # No of points = no of evaluations x no of (points as keys)
            tup = tuple(sorted((pt, wt, cmp) for pt, tup in dual.get_point_dict().items()
                               for (wt, cmp) in tup))
            if len(tup) != 0:
                pts, weights, cmp = zip(*tup)
                alphas = tuple(tuple() for _ in pts)

                try:
                    point_set = point_set_cache[(pts)]
                except KeyError:
                    point_set = PointSet(pts)
                    point_set_cache[(pts)] = point_set
                weights = gem.Indexed(gem.Literal(weights), point_set.indices)
                # TODO: Turn into GEM object?
                # alphas = None
                # cmp = None
                derivs.append((point_set, weights, alphas, cmp))
            else:
                derivs.append(tuple())

            for i in range(1, max_deriv_order+1):
                tup = tuple(sorted((pt, alpha, wt, cmp) for pt, tup in dual.deriv_dict.items()
                            for (wt, alpha, cmp) in tup if sum(alpha) == i))
                if len(tup) != 0:
                    pts, alphas, weights, cmp = zip(*tup)
                    multi_indices = tuple(tuple(chain(*[(i,)*x for i, x in enumerate(alpha)]))
                                          for alpha in alphas)

                    try:
                        point_set = point_set_cache[(pts)]
                    except KeyError:
                        point_set = PointSet(pts)
                        point_set_cache[(pts)] = point_set
                    weights = gem.Indexed(gem.Literal(weights), point_set.indices)
                    # TODO: Turn into GEM object?
                    # multi_indices = None
                    # cmp = None

                    derivs.append((point_set, weights, multi_indices, cmp))
                else:
                    derivs.append(tuple())
            duals.append(tuple(derivs))
        return tuple(duals)

    def dual_evaluation(self, fn, entity=None):
        '''Return code for performing the dual evaluation at the nodes of the
        reference element. Currently only works for point evaluation and quadrature.

        :param fn: Callable that takes in PointSet and returns GEM expression.
        :param entity: the cell entity on which to tabulate for comparing
                       results with FIAT.
        '''
        if entity is None:
            # TODO: Add comparison to FIAT
            pass

        dual_expressions = []   # One for each functional
        expr_cache = {}         # Sharing of evaluation of the expression at points
        # Creates expressions in order of derivative order, extracts and sums alphas
        # and components, then combine with weights
        for dual in self.dual_basis():
            qexprs = gem.Zero()
            for i, deriv in enumerate(dual):
                try:
                    point_set, weights, multi_indices, cmp = deriv
                except ValueError:  # Empty
                    continue

                try:
                    expr = expr_cache[(point_set, multi_indices)]
                except KeyError:
                    expr_grad = fn(point_set, derivative=i)
                    # Extract derivative component
                    # exprs_alpha = gem.ListTensor([gem.partial_indexed(expr_grad, idx) for idx in multi_indices])

                    # Extract derivative and component
                    expr = gem.Zero()
                    for idx, c in zip(multi_indices, cmp):
                        # Insert shape indices between derivative and component?
                        expr = gem.Sum(expr, gem.partial_indexed(gem.partial_indexed(expr_grad, idx), c))

                    print(np.shape(multi_indices), multi_indices)
                    print(np.shape(cmp), cmp)
                    print(weights.shape, weights)
                    # print(exprs_alpha.shape)
                    # x = (gem.Index(), )
                    # expr = gem.index_sum(gem.partial_indexed(exprs_alpha, x), x)

                # Hack to get shape_indices from shape of GEM expression
                # Unsure whether expressions with arguments work
                # Assuming general for all expressions including derivatives
                try:
                    shape_indices
                except NameError:
                    broadcast_shape = len(expr.shape) - len(self.value_shape)
                    shape_indices = tuple(gem.Index() for _ in self.value_shape[:broadcast_shape])

                # Ignore arguments, move to between derivative and component?
                expr = gem.partial_indexed(expr, shape_indices)
                expr_cache[(point_set, multi_indices)] = expr

                # Apply weights
                qexpr = gem.index_sum(weights*expr, point_set.indices)
                # Sum for all derivatives
                qexprs = gem.Sum(qexprs, qexpr)

            assert qexprs.shape == ()
            dual_expressions.append(qexprs)
        ir_shape = gem.ListTensor(dual_expressions)

        return ir_shape

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
                vals = base_values_sympy[const_mask]
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


class DiscontinuousLagrange(ScalarFiatElement):
    def __init__(self, cell, degree):
        super(DiscontinuousLagrange, self).__init__(FIAT.DiscontinuousLagrange(cell, degree))


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
