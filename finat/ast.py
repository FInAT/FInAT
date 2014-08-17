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

    def map_index(self, expr):
        '''Replace indices if they are in the replacements list'''
        try:
            return(self.replacements[expr])
        except IndexError:
            return expr

    def map_foreign(self, expr, *args):

        if isinstance(expr, Recipe):
            return Recipe.replace_indices(self.replacements)
        else:
            return super(_IndexMapper, self).map_foreign(expr, *args)


class _StringifyMapper(StringifyMapper):

    def map_recipe(self, expr, enclosing_prec):
        return self.format("Recipe(%s, %s)",
                           self.rec(expr.indices, PREC_NONE),
                           self.rec(expr.instructions, PREC_NONE))


class Recipe(p.Expression):
    """AST snippets and data corresponding to some form of finite element
    evaluation."""
    def __init__(self, indices, instructions, depends):
        self._indices = tuple(indices)
        self._instructions = instructions
        self._depends = tuple(depends)
        self.children = instructions

    mapper_method = "map_recipe"

    @property
    def indices(self):
        '''The free indices in this :class:`Recipe`.'''

        return self._indices

    @property
    def split_indices(self):
        '''The free indices in this :class:`Recipe` split into dimension
        indices, basis function indices, and point indices.'''

        d = []
        b = []
        p = []

        for i in self._indices:
            if isinstance(i, DimensionIndex):
                d.append(i)
            elif isinstance(i, BasisFunctionIndex):
                b.append(i)
            if isinstance(i, PointIndex):
                p.append(i)

        return map(tuple, (d, b, p))

    @property
    def instructions(self):
        '''The actual instructions making up this :class:`Recipe`.'''

        return self._instructions

    @property
    def depends(self):
        '''The input fields of this :class:`Recipe`.'''

        return self._depends

    def __getinitargs__(self):
        return self._indices, self._instructions, self._depends

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
        return "Recipe(%s, %s)" % (self._indices, self._instructions)

    def replace_indices(self, replacements):
        """Return a copy of this :class:`Recipe` with some of the indices
        substituted."""

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
        return "IndexSum(%s, %s)" % (str([x._str_extent for x in self.children[0]]),
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

    def __init__(self, index, base, expr):
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
