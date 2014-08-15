import pymbolic.primitives as p
from pymbolic.mapper import IdentityMapper
from indices import DimensionIndex, BasisFunctionIndex, PointIndex


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


# Probably Recipe should be a pymbolic.Expression
class Recipe(object):
    """AST snippets and data corresponding to some form of finite element
    evaluation."""
    def __init__(self, indices, instructions, depends):
        self._indices = tuple(indices)
        self._instructions = instructions
        self._depends = tuple(depends)

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


class Delta(p._MultiChildExpression):
    """The Kronecker delta expressed as a ternary operator:

.. math::

    \delta[i_0,i_1]*\mathrm{body}.

:params indices: a sequence of indices.
:params body: an expression.

The body expression will be returned if the values of the indices match. Otherwise 0 will be returned.
    """
    def __init__(self, indices, body):

        self.children = (indices, body)

    def __getinitargs__(self):
        return self.children

    def __str__(self):
        return "Delta(%s, %s)" % self.children
