"""This module defines the additional Pymbolic nodes which are
required to define Finite Element expressions in FInAT.
"""
import pymbolic.primitives as p
try:
    from termcolor import colored
except ImportError:
    def colored(string, color, attrs=[]):
        return string


class FInATSyntaxError(Exception):
    """Exception to raise when users break the rules of the FInAT ast."""


class StringifyMixin(object):
    """Mixin class to set stringification options correctly for pymbolic subclasses."""

    def __str__(self):
        """Use the :meth:`stringifier` to return a human-readable
        string representation of *self*.
        """

        from pymbolic.mapper.stringifier import PREC_NONE
        return self.stringifier()()(self, PREC_NONE, indent=0)

    def stringifier(self):
        from . import mappers
        return mappers._StringifyMapper

    @property
    def name(self):
        if hasattr(self, "_error"):
            return colored(str(self.__class__.__name__), "red", attrs=["bold"])
        else:
            return colored(str(self.__class__.__name__), self._color)

    def set_error(self):
        self._error = True


class Variable(p.Variable):
    """A symbolic variable."""
    def __init__(self, name):
        super(Variable, self).__init__(name)

        self._color = "cyan"

    def set_error(self):
        self._error = True


class Array(Variable):
    """A symbolic variable of known extent."""
    def __init__(self, name, shape):
        super(Array, self).__init__(name)

        self.shape = shape


class Recipe(StringifyMixin, p.Expression):
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
    def __init__(self, indices, body):
        try:
            assert len(indices) == 3
        except:
            raise FInATSyntaxError("Indices must be a triple of tuples")
        self.indices = tuple(indices)
        self.body = body
        self._color = "blue"

    mapper_method = "map_recipe"

    def __getinitargs__(self):
        return self.indices, self.body

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

    def replace_indices(self, replacements):
        """Return a copy of this :class:`Recipe` with some of the indices
        substituted."""

        if not isinstance(replacements, dict):
            replacements = {a: b for (a, b) in replacements}

        from mappers import _IndexMapper
        return _IndexMapper(replacements)(self)


class IndexSum(StringifyMixin, p._MultiChildExpression):
    """A symbolic expression for a sum over one or more indices.

    :param indices: a sequence of indices over which to sum.
    :param body: the expression to sum.
    """
    def __init__(self, indices, body):

        # Inline import to avoid circular dependency.
        from indices import IndexBase
        if isinstance(indices[0], IndexBase):
            indices = tuple(indices)
        else:
            indices = (indices,)

        # Perform trivial simplification of repeated indexsum.
        if isinstance(body, IndexSum):
            indices += body.children[0]
            body = body.children[1]

        self.children = (indices, body)

        self.indices = self.children[0]
        self.body = self.children[1]
        self._color = "blue"

    def __getinitargs__(self):
        return self.children

    mapper_method = "map_index_sum"


class LeviCivita(StringifyMixin, p._MultiChildExpression):
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
        self._color = "blue"

    def __getinitargs__(self):
        return self.children

    mapper_method = "map_index_sum"


class ForAll(StringifyMixin, p._MultiChildExpression):
    """A symbolic expression to indicate that the body will actually be
    evaluated for all of the values of its free indices. This enables
    index simplification to take place.

    :param indices: a sequence of indices to bind.
    :param body: the expression to evaluate.

    """
    def __init__(self, indices, body):

        self.indices = indices
        self.body = body
        self.children = (self.indices, self.body)
        self._color = "blue"

    def __getinitargs__(self):
        return self.children

    mapper_method = "map_for_all"


class Wave(StringifyMixin, p._MultiChildExpression):
    """A symbolic expression with loop-carried dependencies."""

    def __init__(self, var, index, base, update, body):
        self.children = (var, index, base, update, body)
        self._color = "blue"

    def __getinitargs__(self):
        return self.children

    mapper_method = "map_wave"


class Let(StringifyMixin, p._MultiChildExpression):
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

        super(Let, self).__init__((bindings, body))

        self.bindings, self.body = self.children

        self._color = "blue"

    mapper_method = "map_let"


class Delta(StringifyMixin, p._MultiChildExpression):
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
        self._color = "blue"

    def __getinitargs__(self):
        return self.children

    def __str__(self):
        return "Delta(%s, %s)" % self.children

    mapper_method = "map_delta"


class Inverse(StringifyMixin, p.Expression):
    """The inverse of a matrix-valued expression. Where the expression is
    not square, this is the Moore-Penrose pseudo-inverse.

    Where the expression is evaluated at a number of points, the
    inverse will be evaluated pointwise.
    """
    def __init__(self, expression):
        self.expression = expression
        self.children = [expression]
        self._color = "blue"

        super(Inverse, self).__init__()

    def __getinitargs__(self):
        return (self.expression,)

    mapper_method = "map_inverse"


class Det(StringifyMixin, p.Expression):
    """The determinant of a matrix-valued expression."""
    def __init__(self, expression):

        self.expression = expression
        self._color = "blue"

        super(Det, self).__init__()

    def __getinitargs__(self):
        return (self.expression,)

    mapper_method = "map_det"


class Abs(StringifyMixin, p.Expression):
    """The absolute value of an expression."""
    def __init__(self, expression):

        self.expression = expression
        self._color = "blue"

        super(Abs, self).__init__()

    def __getinitargs__(self):
        return (self.expression,)

    mapper_method = "map_abs"


class CompoundVector(StringifyMixin, p.Expression):
    """A vector expression composed by concatenating other expressions."""
    def __init__(self, index, indices, expressions):
        """

        :param index: The free :class:`~.DimensionIndex` created by
        the :class:`CompoundVector`
        :param indices: The sequence of dimension indices of the
        expressions. For scalar components these should be ``None``.
        :param expressions: The sequence of expressions making up
        the compound.

        Each value that `index` takes will be mapped to the corresponding
        value in indices and the matching expression will be evaluated.
        """
        if len(indices) != len(expressions):
            raise FInATSyntaxError("The indices and expressions must be of equal length")

        if sum([i.length for i in indices]) != index.length:
            raise FInATSyntaxError("The length of the compound index must equal "
                                   "the sum of the lengths of the components.")

        super(CompoundVector, self).__init__((index, indices, expressions))

        self.index, self.indices, self.expressions = self.children

        self._color = "blue"

    mapper_method = "map_compound_vector"
