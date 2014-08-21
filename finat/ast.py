"""This module defines the additional Pymbolic nodes which are
required to define Finite Element expressions in FInAT.
"""
import pymbolic.primitives as p
from pymbolic.mapper import IdentityMapper
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE, PREC_CALL
from indices import IndexBase, DimensionIndex, BasisFunctionIndex, PointIndex


class _IndexMapper(IdentityMapper):
    def __init__(self, replacements):
        super(_IndexMapper, self).__init__()

        self.replacements = replacements

    def map_index(self, expr, *args):
        '''Replace indices if they are in the replacements list'''

        try:
            return(self.replacements[expr])
        except KeyError:
            return expr

    def map_recipe(self, expr, *args):
        return expr.__class__(self.rec(expr.indices, *args),
                              self.rec(expr.expression, *args))

    def map_delta(self, expr, *args):
        return expr.__class__(*(self.rec(c, *args) for c in expr.children))

    map_let = map_delta
    map_for_all = map_delta
    map_wave = map_delta
    map_index_sum = map_delta
    map_levi_civita = map_delta


class _StringifyMapper(StringifyMapper):

    def map_subscript(self, expr, enclosing_prec):
        return self.parenthesize_if_needed(
            self.format("%s[%s]",
                        self.rec(expr.aggregate, PREC_CALL),
                        self.join_rec(", ", expr.index, PREC_NONE) if
                        isinstance(expr.index, tuple) else
                        self.rec(expr.index, PREC_NONE)),
            enclosing_prec, PREC_CALL)

    def map_recipe(self, expr, enclosing_prec):
        return self.format("Recipe(%s, %s)",
                           self.rec(expr.indices, PREC_NONE),
                           self.rec(expr.expression, PREC_NONE))

    def map_delta(self, expr, *args):
        return self.format("Delta(%s, %s)",
                           *[self.rec(c, *args) for c in expr.children])

    def map_index_sum(self, expr, *args):
        return self.format("IndexSum((%s), %s)",
                           " ".join(self.rec(c, *args) + "," for c in expr.children[0]),
                           self.rec(expr.children[1], *args))

    def map_levi_civita(self, expr, *args):
        return self.format("LeviCivita(%s)",
                           self.join_rec(", ", expr.children, *args))


class Recipe(p.Expression):
    """AST snippets and data corresponding to some form of finite element
    evaluation.

    A :class:`Recipe` associates an ordered set of indices with an
    expression.

    :param indices: A 3-tuple containing the ordered free indices in the
        expression. The first entry is a tuple of :class:`DimensionIndex`,
        the second is a tuple of :class:`BasisFunctionIndex`, and
        the third is a tuple of :class:`PointIndex`.
        Any of the tuples may be empty.
    :param expression: The expression returned by this :class:`Recipe`.
    """
    def __init__(self, indices, expression):
        try:
            assert len(indices) == 3
        except:
            raise FInATSyntaxError("Indices must be a triple of tuples")
        self.indices = tuple(indices)
        self.expression = expression

    mapper_method = "map_recipe"

    def __getinitargs__(self):
        return self.indices, self.expression

    def __getitem__(self, index):

        replacements = {}

        try:
            for i in range(len(index)):
                if index[i] == slice(None):
                    # Don't touch colon indices.
                    pass
                else:
                    replacements[self.indices[i]] = index[i]
        except TypeError:
            # Index wasn't iterable.
            replacements[self.indices[0]] = index

        return self.replace_indices(replacements)

    def stringifier(self):
        return _StringifyMapper

    def replace_indices(self, replacements):
        """Return a copy of this :class:`Recipe` with some of the indices
        substituted."""

        if not isinstance(replacements, dict):
            replacements = {a: b for (a, b) in replacements}

        return _IndexMapper(replacements)(self)


class IndexSum(p._MultiChildExpression):
    """A symbolic expression for a sum over one or more indices.

    :param indices: a sequence of indices over which to sum.
    :param body: the expression to sum.
    """
    def __init__(self, indices, body):

        if isinstance(indices[0], IndexBase):
            indices = tuple(indices)
        else:
            indices = (indices,)

        self.children = (indices, body)

    def __getinitargs__(self):
        return self.children

    def stringifier(self):
        return _StringifyMapper

    mapper_method = "map_index_sum"


class LeviCivita(p._MultiChildExpression):
    r"""The Levi-Civita symbol expressed as an operator.

    :param free: A tuple of free indices.
    :param bound: A tuple of indices over which to sum.
    :param body: The summand.

    The length of free + bound must be exactly 3. The Levi-Civita
    operator then represents the summation over the bound indices of
    the Levi-Civita symbol times the body. For example in the case of
    two bound indices:

    .. math::
        \mathrm{LeviCivita((\alpha,), (\beta, \gamma), body)} = \sum_{\beta,\gamma}\epsilon_{\alpha,\beta,\gamma} \mathrm{body}

    """
    def __init__(self, free, bound, body):

        self.children = (free, bound, body)

    def __getinitargs__(self):
        return self.children

    def stringifier(self):
        return _StringifyMapper

    mapper_method = "map_index_sum"


class ForAll(p._MultiChildExpression):
    """A symbolic expression to indicate that the body will actually be
    evaluated for all of the values of its free indices. This enables
    index simplification to take place.

    :param indices: a sequence of indices to bind.
    :param body: the expression to evaluate.

    """
    def __init__(self, indices, body):

        self.children = (indices, body)

    def __getinitargs__(self):
        return self.children

    def __str__(self):
        return "ForAll(%s, %s)" % (str([x._str_extent for x in self.children[0]]),
                                   self.children[1])

    mapper_method = "map_for_all"


class Wave(p._MultiChildExpression):
    """A symbolic expression with loop-carried dependencies."""

    def __init__(self, var, index, base, expr):
        pass


class Let(p._MultiChildExpression):
    """A Let expression enables local variable bindings in an
expression. This feature is lifted more or less directly from
Scheme.

:param bindings: A tuple of pairs. The first entry in each pair is a
    :class:`pymbolic.Variable` to be defined as the second entry, which
    must be an expression.
:param body: The expression making up the body of the expression. The
    value of the Let expression is the value of this expression.

    """

    def __init__(self, bindings, body):
        try:
            for b in bindings:
                assert len(b) == 2
        except:
            raise FInATSyntaxError("Let bindings must be a tuple of pairs")

        super(Wave, self).__init__((bindings, body))

    def __str__(self):
        return "Let(%s)" % self.children

    mapper_method = "map_let"


class Delta(p._MultiChildExpression):
    """The Kronecker delta expressed as a ternary operator:

.. math::

    \mathrm{Delta((i, j), body)} = \delta_{ij}*\mathrm{body}.

:param indices: a sequence of indices.
:param body: an expression.

The body expression will be returned if the values of the indices
match. Otherwise 0 will be returned.

    """
    def __init__(self, indices, body):
        if len(indices) != 2:
            raise FInATSyntaxError(
                "Delta statement requires exactly two indices")

        super(Delta, self).__init__((indices, body))

    def __getinitargs__(self):
        return self.children

    def __str__(self):
        return "Delta(%s, %s)" % self.children

    mapper_method = "map_delta"

    def stringifier(self):
        return _StringifyMapper


class FInATSyntaxError(Exception):
    """Exception raised when the syntax rules of the FInAT ast are violated."""
    pass
