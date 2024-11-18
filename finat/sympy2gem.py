from functools import singledispatch, reduce

import numpy
import sympy
try:
    import symengine
except ImportError:
    class Mock:
        def __getattribute__(self, name):
            return Mock
    symengine = Mock()

import gem


@singledispatch
def sympy2gem(node, self):
    raise AssertionError("sympy/symengine node expected, got %s" % type(node))


@sympy2gem.register(sympy.Expr)
@sympy2gem.register(symengine.Expr)
def sympy2gem_expr(node, self):
    raise NotImplementedError("no handler for sympy/symengine node type %s" % type(node))


@sympy2gem.register(sympy.Add)
@sympy2gem.register(symengine.Add)
def sympy2gem_add(node, self):
    return reduce(gem.Sum, map(self, node.args))


@sympy2gem.register(sympy.Mul)
@sympy2gem.register(symengine.Mul)
def sympy2gem_mul(node, self):
    return reduce(gem.Product, map(self, node.args))


@sympy2gem.register(sympy.Pow)
@sympy2gem.register(symengine.Pow)
def sympy2gem_pow(node, self):
    return gem.Power(*map(self, node.args))


@sympy2gem.register(sympy.logic.boolalg.BooleanTrue)
@sympy2gem.register(sympy.logic.boolalg.BooleanFalse)
@sympy2gem.register(bool)
def sympy2gem_boolean(node, self):
    return gem.Literal(bool(node))


@sympy2gem.register(sympy.Integer)
@sympy2gem.register(symengine.Integer)
@sympy2gem.register(int)
def sympy2gem_integer(node, self):
    return gem.Literal(int(node))


@sympy2gem.register(sympy.Float)
@sympy2gem.register(symengine.Float)
@sympy2gem.register(float)
def sympy2gem_float(node, self):
    return gem.Literal(float(node))


@sympy2gem.register(sympy.Symbol)
@sympy2gem.register(symengine.Symbol)
def sympy2gem_symbol(node, self):
    return self.bindings[node]


@sympy2gem.register(sympy.Rational)
@sympy2gem.register(symengine.Rational)
def sympy2gem_rational(node, self):
    return gem.Division(*(map(self, node.as_numer_denom())))


@sympy2gem.register(sympy.Abs)
@sympy2gem.register(symengine.Abs)
def sympy2gem_abs(node, self):
    return gem.MathFunction("abs", *map(self, node.args))


@sympy2gem.register(sympy.Not)
@sympy2gem.register(symengine.Not)
def sympy2gem_not(node, self):
    return gem.LogicalNot(*map(self, node.args))


@sympy2gem.register(sympy.Or)
@sympy2gem.register(symengine.Or)
def sympy2gem_or(node, self):
    return reduce(gem.LogicalOr, map(self, node.args))


@sympy2gem.register(sympy.And)
@sympy2gem.register(symengine.And)
def sympy2gem_and(node, self):
    return reduce(gem.LogicalAnd, map(self, node.args))


@sympy2gem.register(sympy.Eq)
@sympy2gem.register(symengine.Eq)
def sympy2gem_eq(node, self):
    return gem.Comparison("==", *map(self, node.args))


@sympy2gem.register(sympy.Gt)
def sympy2gem_gt(node, self):
    return gem.Comparison(">", *map(self, node.args))


@sympy2gem.register(sympy.Ge)
def sympy2gem_ge(node, self):
    return gem.Comparison(">=", *map(self, node.args))


@sympy2gem.register(sympy.Lt)
@sympy2gem.register(symengine.Lt)
def sympy2gem_lt(node, self):
    return gem.Comparison("<", *map(self, node.args))


@sympy2gem.register(sympy.Le)
@sympy2gem.register(symengine.Le)
def sympy2gem_le(node, self):
    return gem.Comparison("<=", *map(self, node.args))


@sympy2gem.register(sympy.Piecewise)
@sympy2gem.register(symengine.Piecewise)
def sympy2gem_conditional(node, self):
    expr = None
    pieces = []
    for v, c in node.args:
        if isinstance(c, (bool, numpy.bool, sympy.logic.boolalg.BooleanTrue)) and c:
            expr = self(v)
            break
        pieces.append((v, c))
    if expr is None:
        expr = gem.Literal(float("nan"))
    for v, c in reversed(pieces):
        expr = gem.Conditional(self(c), self(v), expr)
    return expr


@sympy2gem.register(sympy.ITE)
def sympy2gem_ifthenelse(node, self):
    return gem.Conditional(*map(self, node.args))
