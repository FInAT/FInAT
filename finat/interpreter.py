"""An interpreter for FInAT recipes. This is not expected to be
performant, but rather to provide a test facility for FInAT code."""
import pymbolic.primitives as p
from pymbolic.mapper.evaluator import FloatEvaluationMapper, UnknownVariableError
from ast import IndexSum, ForAll, LeviCivita, FInATSyntaxError
from indices import TensorPointIndex
import numpy as np
import copy


class FinatEvaluationMapper(FloatEvaluationMapper):

    def __init__(self, context={}):
        """
        :arg context: a mapping from variable names to values
        """
        super(FinatEvaluationMapper, self).__init__(context)

        self.indices = {}

        # Storage for wave variables while they are out of scope.
        self.wave_vars = {}

    def _as_range(self, e):
        """Convert a slice to a range. If the range has expressions as bounds,
        evaluate them.
        """

        return range(int(self.rec(e.start or 0)),
                     int(self.rec(e.stop)),
                     int(self.rec(e.step or 1)))

    def map_variable(self, expr):
        try:
            var = self.context[expr.name]
            if isinstance(var, p.Expression):
                return self.rec(var)
            else:
                return var
        except KeyError:
            expr.set_error()
            raise UnknownVariableError(expr.name)

    def map_recipe(self, expr):
        """Evaluate expr for all values of free indices"""

        d, b, p = expr.indices

        free_indices = [i for i in d + b + p if i not in self.indices]

        try:
            forall = ForAll(free_indices, expr.body)
            return self.rec(forall)
        except:
            if hasattr(forall, "_error"):
                expr.set_error()
            raise

    def map_index_sum(self, expr):

        indices, body = expr.children
        expr_in = expr

        # Sum over multiple indices recursively.
        if len(indices) > 1:
            expr = IndexSum(indices[1:], body)
        else:
            expr = body

        idx = indices[0]

        if idx in self.indices:
            expr_in.set_error()
            idx.set_error()
            raise FInATSyntaxError("Attempting to bind the name %s which is already bound" % idx)

        e = idx.extent

        total = 0.0

        self.indices[idx] = None

        for i in self._as_range(e):
            self.indices[idx] = i
            try:
                total += self.rec(expr)
            except:
                if hasattr(expr, "_error"):
                    expr_in.set_error()
                raise

        self.indices.pop(idx)

        return total

    def map_index(self, expr):

        try:
            return self.indices[expr]
        except KeyError:
            expr.set_error()
            raise FInATSyntaxError("Access to unbound variable name %s." % expr)

    def map_for_all(self, expr):

        indices, body = expr.children
        expr_in = expr

        # Deal gracefully with the zero index case.
        if not indices:
            return self.rec(body)

        # Execute over multiple indices recursively.
        if len(indices) > 1:
            expr = ForAll(indices[1:], body)
        # Expand tensor indices
        elif isinstance(indices[0], TensorPointIndex):
            indices = indices[0].factors
            expr = ForAll(indices[1:], body)
        else:
            expr = body

        idx = indices[0]

        if idx in self.indices:
            expr_in.set_error()
            idx.set_error()
            raise FInATSyntaxError(
                "Attempting to bind the name %s which is already bound" % idx)

        e = idx.extent

        total = []
        for i in self._as_range(e):
            self.indices[idx] = i
            try:
                total.append(self.rec(expr))
            except:
                if hasattr(expr, "_error"):
                    expr_in.set_error()
                raise

        self.indices.pop(idx)

        return np.array(total)

    def map_compound_vector(self, expr):

        (index, indices, bodies) = expr.children

        if index not in self.indices:
            expr.set_error()
            index.set_error()
            raise FInATSyntaxError(
                "Compound vector depends on %s, which is not in scope" % index)

        alpha = self.indices[index]

        for idx, body in zip(indices, bodies):
            if alpha < idx.length:
                if idx in self.indices:
                    raise FInATSyntaxError(
                        "Attempting to bind the name %s which is already bound" % idx)
                self.indices[idx] = self._as_range(idx)[alpha]
                result = self.rec(body)
                self.indices.pop(idx)
                return result
            else:
                alpha -= idx.length

        raise FInATSyntaxError("Compound index %s out of bounds" % index)

    def map_wave(self, expr):

        (var, index, base, update, body) = expr.children

        if index not in self.indices:
            expr.set_error()
            index.set_error()
            raise FInATSyntaxError(
                "Wave variable depends on %s, which is not in scope" % index)

        try:
            self.context[var.name] = self.wave_vars[var.name]
            self.context[var.name] = self.rec(update)
        except KeyError:
            # We're at the start of the loop over index.
            assert self.rec(index) == (index.extent.start or 0)
            self.context[var.name] = self.rec(base)

        self.wave_vars[var.name] = self.context[var.name]

        # Execute the body.
        result = self.rec(body)

        # Remove the wave variable from scope.
        self.context.pop(var.name)
        if self.rec(index) >= self.rec(index.extent.stop) - 1:
            self.wave_vars.pop(var.name)

        return result

    def map_let(self, expr):

        for var, value in expr.bindings:
            if var in self.context:
                expr.set_error()
                var.set_error()
                raise FInATSyntaxError("Let variable %s was already in scope."
                                       % var.name)
            self.context[var.name] = self.rec(value)

        result = self.rec(expr.body)

        for var, value in expr.bindings:
            self.context.pop(var.name)

        return result

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

        expr.set_error()
        raise NotImplementedError

    def map_det(self, expr):

        return np.linalg.det(self.rec(expr.expression))

    def map_abs(self, expr):

        return abs(self.rec(expr.expression))


def evaluate(expression, context={}, kernel_data=None):
    """Take a FInAT expression and a set of definitions for undefined
    variables in the expression, and optionally the current kernel
    data. Evaluate the expression returning a Float or numpy array
    according to the size of the expression.
    """

    if kernel_data:
        context = copy.copy(context)
        for var in kernel_data.static.values():
            context[var[0].name] = var[1]()

    try:
        return FinatEvaluationMapper(context)(expression)
    except:
        print expression
        raise
