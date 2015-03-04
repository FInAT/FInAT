from pymbolic.mapper import IdentityMapper as IM
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE
from pymbolic.mapper import WalkMapper as WM
from pymbolic.mapper.graphviz import GraphvizMapper as GVM
from .indices import IndexBase
from .ast import Recipe, ForAll, IndexSum
try:
    from termcolor import colored
except ImportError:
    def colored(string, color, attrs=[]):
        return string


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

    def map_inverse(self, expr, *args):
        return expr.__class__(self.rec(expr.expression, *args))

    map_let = map_delta
    map_for_all = map_delta
    map_wave = map_delta
    map_index_sum = map_delta
    map_levi_civita = map_delta
    map_compound_vector = map_delta
    map_det = map_inverse
    map_abs = map_inverse


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

    map_for_all = map_recipe

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
        if hasattr(expr, "_error"):
            return colored(str(expr), "red", attrs=["bold"])
        else:
            return colored(str(expr), expr._color)

    def map_wave(self, expr, enclosing_prec, indent=None, *args, **kwargs):
        if indent is None or enclosing_prec is not PREC_NONE:
            fmt = expr.name + "(%s %s) "
        else:
            oldidt = " " * indent
            indent += 4
            idt = " " * indent
            fmt = expr.name + "(%s\n" + idt + "%s\n" + oldidt + ")"

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

    map_abs = map_det

    def map_compound_vector(self, expr, *args, **kwargs):
        return self.format(expr.name + "(%s)",
                           self.join_rec(", ", expr.children, *args, **kwargs))

    def map_variable(self, expr, enclosing_prec, *args, **kwargs):
        if hasattr(expr, "_error"):
            return colored(str(expr.name), "red", attrs=["bold"])
        else:
            try:
                return colored(expr.name, expr._color)
            except AttributeError:
                return colored(expr.name, "cyan")


class WalkMapper(WM):
    def __init__(self):
        super(WalkMapper, self).__init__()

    def map_recipe(self, expr, *args, **kwargs):
        if not self.visit(expr, *args, **kwargs):
            return
        for indices in expr.indices:
            for index in indices:
                self.rec(index, *args, **kwargs)
        self.rec(expr.body, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    def map_index(self, expr, *args, **kwargs):
        if not self.visit(expr, *args, **kwargs):
            return

        # I don't want to recur on the extent.  That's ugly.

        self.post_visit(expr, *args, **kwargs)

    def map_index_sum(self, expr, *args, **kwargs):
        if not self.visit(expr, *args, **kwargs):
            return
        for index in expr.indices:
            self.rec(index, *args, **kwargs)
        self.rec(expr.body, *args, **kwargs)
        self.post_visit(expr, *args, **kwargs)

    map_delta = map_index_sum
    map_let = map_index_sum
    map_for_all = map_index_sum
    map_wave = map_index_sum
    map_levi_civita = map_index_sum
    map_inverse = map_index_sum
    map_det = map_index_sum
    map_compound_vector = map_index_sum


class GraphvizMapper(WalkMapper, GVM):
    pass


class BindingMapper(IdentityMapper):
    """A mapper that binds free indices in recipes using ForAlls."""

    def __init__(self, kernel_data):
        """
        :arg context: a mapping from variable names to values
        """
        super(BindingMapper, self).__init__()
        self.bound_above = set()
        self.bound_below = set()

    def map_recipe(self, expr):
        body = self.rec(expr.body)

        d, b, p = expr.indices
        free_indices = tuple([i for i in d + b + p
                              if i not in self.bound_below and
                              i not in self.bound_above])

        if len(free_indices) > 0:
            expr = Recipe(expr.indices, ForAll(free_indices, body))

        return expr

    def map_index_sum(self, expr):
        indices = expr.indices
        for idx in indices:
            self.bound_above.add(idx)
        body = self.rec(expr.body)
        for idx in indices:
            self.bound_above.remove(idx)
            self.bound_below.add(idx)
        return IndexSum(indices, body)
