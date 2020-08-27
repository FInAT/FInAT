from functools import reduce, singledispatch

import symengine
import sympy
import ufl
from gem.node import Memoizer


@singledispatch
def _sympy2ufl(node, self):
    raise AssertionError("sympy/symengine node expected, got %s" % type(node))


@_sympy2ufl.register(sympy.Expr)
@_sympy2ufl.register(symengine.Expr)
def sympy2ufl_expr(node, self):
    raise NotImplementedError("no handler for sympy/symengine node type %s" % type(node))


@_sympy2ufl.register(sympy.Add)
@_sympy2ufl.register(symengine.Add)
def sympy2ufl_add(node, self):
    return reduce(ufl.classes.Sum, map(self, node.args))


@_sympy2ufl.register(sympy.Mul)
@_sympy2ufl.register(symengine.Mul)
def sympy2ufl_mul(node, self):
    return reduce(ufl.classes.Product, map(self, node.args))


@_sympy2ufl.register(sympy.Pow)
@_sympy2ufl.register(symengine.Pow)
def sympy2ufl_pow(node, self):
    return ufl.classes.Power(*map(self, node.args))


@_sympy2ufl.register(sympy.Integer)
@_sympy2ufl.register(symengine.Integer)
@_sympy2ufl.register(int)
def sympy2ufl_integer(node, self):
    return ufl.classes.IntValue(int(node))


@_sympy2ufl.register(sympy.Float)
@_sympy2ufl.register(symengine.Float)
@_sympy2ufl.register(float)
def sympy2ufl_float(node, self):
    return ufl.classes.FloatValue(float(node))


@_sympy2ufl.register(sympy.Symbol)
@_sympy2ufl.register(symengine.Symbol)
def sympy2ufl_symbol(node, self):
    return self.bindings[node]


@_sympy2ufl.register(sympy.Rational)
@_sympy2ufl.register(symengine.Rational)
def sympy2ufl_rational(node, self):
    return ufl.classes.Division(*(map(self, node.as_numer_denom())))


@_sympy2ufl.register(sympy.Array)
def sympy2ufl_array(node, self):
    import numpy
    vals, shape = node.args
    return ufl.as_tensor(numpy.asarray([self(v) for v in vals]).reshape(shape))

@_sympy2ufl.register(sympy.Piecewise)
def sympy2ufl_piecewise(node, self):
    import numpy
    def recurse(exprs, conditions):
        try:
            expr, = exprs
            condition, = conditions
            if type(condition) is bool:
                # Sympy claims to drop known False parts.
                assert condition
                return expr
            else:
                return ufl.conditional(condition, expr,
                                       ufl.classes.FloatValue(numpy.nan))
        except ValueError:
            expr, *exprs = exprs
            condition, *conditions = conditions
            return ufl.conditional(condition, expr, recurse(exprs, conditions))
    return recurse(*zip(*map(self, node.args)))

@_sympy2ufl.register(sympy.core.numbers.Zero)
def sympy2ufl_zero(node, self):
    return ufl.zero()
@_sympy2ufl.register(sympy.functions.elementary.piecewise.ExprCondPair)
def sympy2ufl_exprcondpair(node, self):
    return tuple(map(self, node.args))

@_sympy2ufl.register(sympy.core.relational.StrictLessThan)
def sympy2ufl_lt(node, self):
    return ufl.classes.LT(*(map(self, node.args)))

@_sympy2ufl.register(sympy.core.relational.StrictGreaterThan)
def sympy2ufl_gt(node, self):
    return ufl.classes.GT(*(map(self, node.args)))

from sympy.logic import boolalg
@_sympy2ufl.register(boolalg.BooleanTrue)
def sympy2ufl_bool(node, self):
    return True

def sympy2ufl(expr, bindings={}):
    mapper = Memoizer(_sympy2ufl)
    mapper.bindings = bindings
    return mapper(expr)
