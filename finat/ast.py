"""This module defines the additional Pymbolic nodes which are
required to define Finite Element expressions in FInAT.
"""
import pymbolic.primitives as p
from pymbolic.mapper import IdentityMapper as IM
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE
from indices import IndexBase
try:
    from termcolor import colored
except ImportError:
    colored = lambda string, color: string


class FInATSyntaxError(Exception):
    """Exception to raise when users break the rules of the FInAT ast."""


class IdentityMapper(IM):
    def __init__(self):
        super(IdentityMapper, self).__init__()

    def map_recipe(self, expr, *args):
        return expr.__class__(self.rec(expr.indices, *args),
                              self.rec(expr.body, *args))

    def map_index(self, expr, *args):
        return expr

    def map_delta(self, expr, *args):
        return expr.__class__(*(self.rec(c, *args) for c in expr.children))

    map_let = map_delta
    map_for_all = map_delta
    map_wave = map_delta
    map_index_sum = map_delta
    map_levi_civita = map_delta
    map_inverse = map_delta
    map_det = map_delta


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


class _StringifyMapper(StringifyMapper):

    def map_recipe(self, expr, enclosing_prec, indent=None, *args, **kwargs):
        if indent is None:
            fmt = expr.name + "(%s, %s)"
        else:
            oldidt = " " * indent
            indent += 4
            idt = " " * indent
            fmt = expr.name + "(%s,\n" + idt + "%s\n" + oldidt + ")"

        return self.format(fmt,
                           self.rec(expr.indices, PREC_NONE, indent=indent, *args, **kwargs),
                           self.rec(expr.body, PREC_NONE, indent=indent, *args, **kwargs))

    def map_let(self, expr, enclosing_prec, indent=None, *args, **kwargs):
        if indent is None:
            fmt = expr.name + "(%s, %s)"
            inner_indent = None
        else:
            oldidt = " " * indent
            indent += 4
            inner_indent = indent + 4
            inner_idt = " " * inner_indent
            idt = " " * indent
            fmt = expr.name + "(\n" + inner_idt + "%s,\n" + idt + "%s\n" + oldidt + ")"

        return self.format(fmt,
                           self.rec(expr.bindings, PREC_NONE, indent=inner_indent, *args, **kwargs),
                           self.rec(expr.body, PREC_NONE, indent=indent, *args, **kwargs))

    def map_delta(self, expr, *args, **kwargs):
        return self.format(expr.name + "(%s, %s)",
                           *[self.rec(c, *args, **kwargs) for c in expr.children])

    def map_index(self, expr, *args, **kwargs):
        return colored(str(expr), expr._color)

    def map_wave(self, expr, enclosing_prec, indent=None, *args, **kwargs):
        if indent is None or enclosing_prec is not PREC_NONE:
            fmt = expr.name + "(%s, %s) "
        else:
            oldidt = " " * indent
            indent += 4
            idt = " " * indent
            fmt = expr.name + "(%s,\n" + idt + "%s\n" + oldidt + ")"

        return self.format(fmt,
                           " ".join(self.rec(c, PREC_NONE, *args, **kwargs) + "," for c in expr.children[:-1]),
                           self.rec(expr.children[-1], PREC_NONE, indent=indent, *args, **kwargs))

    def map_index_sum(self, expr, enclosing_prec, indent=None, *args, **kwargs):
        if indent is None or enclosing_prec is not PREC_NONE:
            fmt = expr.name + "((%s), %s) "
        else:
            oldidt = " " * indent
            indent += 4
            idt = " " * indent
            fmt = expr.name + "((%s),\n" + idt + "%s\n" + oldidt + ")"

        return self.format(fmt,
                           " ".join(self.rec(c, PREC_NONE, *args, **kwargs) + "," for c in expr.children[0]),
                           self.rec(expr.children[1], PREC_NONE, indent=indent, *args, **kwargs))

    def map_levi_civita(self, expr, *args, **kwargs):
        return self.format(expr.name + "(%s)",
                           self.join_rec(", ", expr.children, *args, **kwargs))

    def map_inverse(self, expr, *args, **kwargs):
        return self.format(expr.name + "(%s)",
                           self.rec(expr.expression, *args, **kwargs))

    def map_det(self, expr, *args, **kwargs):
        return self.format(expr.name + "(%s)",
                           self.rec(expr.expression, *args, **kwargs))

    def map_variable(self, expr, enclosing_prec, *args, **kwargs):
        try:
            return colored(expr.name, expr._color)
        except AttributeError:
            return colored(expr.name, "cyan")



class StringifyMixin(object):
    """Mixin class to set stringification options correctly for pymbolic subclasses."""

    def __str__(self):
        """Use the :meth:`stringifier` to return a human-readable
        string representation of *self*.
        """

        from pymbolic.mapper.stringifier import PREC_NONE
        return self.stringifier()()(self, PREC_NONE, indent=0)

    def stringifier(self):
        return _StringifyMapper

    @property
    def name(self):
        return colored(str(self.__class__.__name__), self._color)


class Array(p.Variable):
    """A pymbolic variable of known extent."""
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

        return _IndexMapper(replacements)(self)


class IndexSum(StringifyMixin, p._MultiChildExpression):
    """A symbolic expression for a sum over one or more indices.

    :param indices: a sequence of indices over which to sum.
    :param body: the expression to sum.
    """
    def __init__(self, indices, body):

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

        self.children = (indices, body)
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
        self._color = "blue"

    mapper_method = "map_inverse"


class Det(StringifyMixin, p.Expression):
    """The determinant of a matrix-valued expression.

    Where the expression is evaluated at a number of points, the
    inverse will be evaluated pointwise.
    """
    def __init__(self, expression):

        self.expression = expression
        self._color = "blue"

    mapper_method = "map_det"
