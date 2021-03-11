from functools import singledispatch, reduce

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
