from functools import singledispatch, reduce

import sympy

import gem


@singledispatch
def sympy2gem(node, self):
    raise AssertionError("sympy node expected, got %s" % type(node))


@sympy2gem.register(sympy.Expr)
def sympy2gem_expr(node, self):
    raise NotImplementedError("no handler for sympy node type %s" % type(node))


@sympy2gem.register(sympy.Add)
def sympy2gem_add(node, self):
    return reduce(gem.Sum, map(self, node.args))


@sympy2gem.register(sympy.Mul)
def sympy2gem_mul(node, self):
    return reduce(gem.Product, map(self, node.args))


@sympy2gem.register(sympy.Pow)
def sympy2gem_pow(node, self):
    return gem.Power(*map(self, node.args))


@sympy2gem.register(sympy.Integer)
@sympy2gem.register(int)
def sympy2gem_integer(node, self):
    return gem.Literal(node)


@sympy2gem.register(sympy.Float)
@sympy2gem.register(float)
def sympy2gem_float(node, self):
    return gem.Literal(node)


@sympy2gem.register(sympy.Symbol)
def sympy2gem_symbol(node, self):
    return self.bindings[node]


@sympy2gem.register(sympy.Rational)
def sympy2gem_rational(node, self):
    return gem.Division(self(node.numerator()), self(node.denominator()))
