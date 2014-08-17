"""This module defines the additional Pymbolic nodes which are
required to define Finite Element expressions in FInAT.
"""
import pymbolic.primitives as p
from pymbolic.mapper import IdentityMapper
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE
from indices import IndexBase, DimensionIndex, BasisFunctionIndex, PointIndex


class _IndexMapper(IdentityMapper):
    def __init__(self, replacements):
        super(_IndexMapper, self).__init__()

        self.replacements = replacements

    def map_index(self, expr, *args):
        '''Replace indices if they are in the replacements list'''
        print expr, self.replacements
        try:
            return(self.replacements[expr])
        except KeyError:
            return expr

    def map_recipe(self, expr, *args):
        return Recipe(self.rec(expr.indices),
                      self.rec(expr.expression))


class _StringifyMapper(StringifyMapper):

    def map_recipe(self, expr, enclosing_prec):
        return self.format("Recipe(%s, %s)",
                           self.rec(expr.indices, PREC_NONE),
                           self.rec(expr.expression, PREC_NONE))


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

    def __str__(self):
        return "Recipe(%s, %s)" % (self.indices, self.expression)

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

    def __str__(self):
        return "IndexSum(%s, %s)" % (tuple(map(str, self.children[0])),
                                     self.children[1])


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


class Delta(p._MultiChildExpression):
    """The Kronecker delta expressed as a ternary operator:

.. math::

    \delta[i_0,i_1]*\mathrm{body}.

:param indices: a sequence of indices.
:param body: an expression.

The body expression will be returned if the values of the indices
match. Otherwise 0 will be returned.

    """
    def __init__(self, indices, body):
        if len(indices != 2):
            raise FInATSyntaxError(
                "Delta statement requires exactly two indices")

        super(Delta, self).__init__((indices, body))

    def __getinitargs__(self):
        return self.children

    def __str__(self):
        return "Delta(%s, %s)" % self.children


class FInATSyntaxError(Exception):
    """Exception raised when the syntax rules of the FInAT ast are violated."""
    pass
