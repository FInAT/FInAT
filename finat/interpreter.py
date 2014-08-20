"""An interpreter for FInAT recipes. This is not expected to be
performant, but rather to provide a test facility for FInAT code."""
import pymbolic.primitives as p
from pymbolic.mapper.evaluator import FloatEvaluationMapper, UnknownVariableError
from ast import IndexSum, ForAll, LeviCivita
import numpy as np
import copy

def _as_range(e):
    """Convert a slice to a range."""

    return range(e.start or 0, e.stop, e.step or 1)


class FinatEvaluationMapper(FloatEvaluationMapper):

    def __init__(self, context={}):
        """
        :arg context: a mapping from variable names to values
        """
        super(FinatEvaluationMapper, self).__init__(context)

        self.indices = {}

    def map_variable(self, expr):
        try:
            var = self.context[expr.name]
            if isinstance(var, p.Expression):
                return self.rec(var)
            else:
                return var
        except KeyError:
            raise UnknownVariableError(expr.name)

    def map_recipe(self, expr):
        """Evaluate expr for all values of free indices"""

        d, b, p = expr.indices
        body = expr.expression

        return self.rec(ForAll(d+b+p, body))

    def map_index_sum(self, expr):

        indices, body = expr.children

        # Sum over multiple indices recursively.
        if len(indices) > 1:
            expr = IndexSum(indices[1:], body)

        idx = indices[0]

        e = idx.extent

        total = 0.0
        for i in _as_range(e):
            self.indices[idx] = i
            total += self.rec(expr)

        self.indices.pop(idx)

        return total

    def map_index(self, expr):

        return self.indices[expr]

    def map_for_all(self, expr):

        indices, body = expr.children

        # Execute over multiple indices recursively.
        if len(indices) > 1:
            expr = IndexSum(indices[1:], body)

        idx = indices[0]

        e = idx.extent

        total = []
        for i in _as_range(e):
            self.indices[idx] = i
            total.append(self.rec(expr))

        self.indices.pop(idx)

        return np.array(total)

    def map_levi_civita(self, expr):

        free, bound, body = expr.children

        if len(bound) == 3:
            return self.rec(IndexSum(bound[:1],
                                     LeviCivita(bound[:1], bound[1:], body)))
        elif len(bound) == 2:

            self.indices[bound[0]] = (self.indices[free[0]] + 1) % 3
            self.indices[bound[1]] = (self.indices[free[0]] + 2) % 3
            tmp = self.rec(body)
            self.indices[bound[1]] = (self.indices[free[0]] + 1) % 3
            self.indices[bound[0]] = (self.indices[free[0]] + 2) % 3
            tmp -= self.rec(body)

            self.indices.pop(bound[0])
            self.indices.pop(bound[1])

            return tmp

        elif len(bound) == 1:

            i = self.indices[free[0]]
            j = self.indices[free[1]]

            if i == j:
                return 0
            elif j == (i + 1) % 3:
                k = i + 2 % 3
                sign = 1
            elif j == (i + 2) % 3:
                k = i + 1 % 3
                sign = -1

            self.indices[bound[0]] = k
            return sign * self.rec(body)

        elif len(bound) == 0:

            eijk = np.zeros((3, 3, 3))
            eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
            eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

            i = self.indices[free[0]]
            j = self.indices[free[1]]
            k = self.indices[free[1]]
            sign = eijk[i, j, k]

            if sign != 0:
                return sign * self.rec(body)
            else:
                return 0

        raise NotImplementedError


def evaluate(expression, context={}, kernel_data=None):
    """Take a FInAT expression and a set of definitions for undefined
    variables in the expression, and optionally the current kernel
    data. Evaluate the expression returning a Float or numpy array
    according to the size of the expression.
    """

    if kernel_data:
        context = copy.copy(context)
        for var in kernel_data.static.values():
            context[var[0]] = var[1]

    return FinatEvaluationMapper(context)(expression)
