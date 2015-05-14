from collections import deque
from pymbolic.mapper import IdentityMapper as IM
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE
from pymbolic.mapper import WalkMapper as WM
from pymbolic.mapper.graphviz import GraphvizMapper as GVM
from pymbolic.primitives import Product, Sum, flattened_product, flattened_sum
from .indices import IndexBase, TensorIndex, PointIndexBase, BasisFunctionIndexBase,\
    DimensionIndex
from .ast import Recipe, ForAll, IndexSum, Let, Variable, Delta, CompoundVector
try:
    from termcolor import colored
except ImportError:
    def colored(string, color, attrs=None):
        return string


class IdentityMapper(IM):
    def __init__(self):
        super(IdentityMapper, self).__init__()

    def map_recipe(self, expr, *args, **kwargs):
        return expr.__class__(self.rec(expr.indices, *args, **kwargs),
                              self.rec(expr.body, *args, **kwargs),
                              expr._transpose)

    def map_index(self, expr, *args, **kwargs):
        return expr

    def map_delta(self, expr, *args, **kwargs):
        return expr.__class__(*tuple(self.rec(c, *args, **kwargs)
                                     for c in expr.children))

    def map_inverse(self, expr, *args, **kwargs):
        return expr.__class__(self.rec(expr.expression, *args, **kwargs))

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

    def map_index(self, expr, *args, **kwargs):
        '''Replace indices if they are in the replacements list'''

        try:
            return(self.replacements[expr])
        except KeyError:
            return expr

    def map_compound_vector(self, expr, *args, **kwargs):
        # Symbolic replacement of indices on a CompoundVector Just
        # Works. However replacing the compound vector index with a
        # number should collapse the CompoundVector.

        if expr.index in self.replacements and\
           not isinstance(self.replacements[expr.index], IndexBase):
            # Work out which subvector we are in and what the index value is.
            i = expr.index
            val = self.replacements[expr.index]

            pos = (val - (i.start or 0)) / (i.step or 1)
            assert pos <= i.length

            for subindex, body in zip(expr.indices, expr.body):
                if pos < subindex.length:
                    sub_i = pos * (subindex.step or 1) + (subindex.start or 0)
                    self.replacements[subindex] = sub_i
                    result = self.rec(body, *args, **kwargs)
                    self.replacements.pop(subindex)
                    return result
                else:
                    pos -= subindex.length

            raise ValueError("Illegal index value.")
        else:
            return super(_IndexMapper, self).map_compound_vector(expr, *args, **kwargs)


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
    map_for_all = map_index_sum
    map_wave = map_index_sum
    map_levi_civita = map_index_sum
    map_inverse = map_index_sum
    map_det = map_index_sum
    map_compound_vector = map_index_sum


class IndicesMapper(WalkMapper):
    """Label an AST with the indices which occur below each node."""

    def __init__(self):
        self._index_stack = [set()]

    def visit(self, expr, *args, **kwargs):
        # Put a new index frame onto the stack.
        self._index_stack.append(set())
        return True

    def post_visit(self, expr, *args, **kwargs):
        # The frame contains any indices we directly saw:
        expr._indices_below = tuple(self._index_stack.pop())

        if isinstance(expr, IndexBase):
            expr._indices_below += expr

        self._index_stack[-1].union(expr._indices_below)


class GraphvizMapper(WalkMapper, GVM):
    pass


class BindingMapper(IdentityMapper):
    """A mapper that binds free indices in recipes using ForAlls."""

    def __init__(self, kernel_data):
        """
        :arg context: a mapping from variable names to values
        """
        super(BindingMapper, self).__init__()

    def map_recipe(self, expr, bound_above=None, bound_below=None):
        if bound_above is None:
            bound_above = set()
        if bound_below is None:
            bound_below = deque()

        body = self.rec(expr.body, bound_above, bound_below)

        d, b, p = expr.indices
        recipe_indices = tuple([i for i in d + b + p
                                if i not in bound_above])
        free_indices = tuple(set([i for i in recipe_indices
                                  if i not in bound_below]))

        bound_below.extendleft(reversed(free_indices))
        # Calculate the permutation from the order of loops actually
        # employed to the ordering of indices in the Recipe.
        try:
            def expand_tensors(indices):
                result = []
                if indices:
                    for i in indices:
                        try:
                            result += i.factors
                        except AttributeError:
                            result.append(i)
                return result

            tmp = expand_tensors(recipe_indices)
            transpose = [tmp.index(i) for i in expand_tensors(bound_below)]
        except ValueError:
            print "recipe_indices", recipe_indices
            print "missing index", i
            i.set_error()
            raise

        if len(free_indices) > 0:
            expr = Recipe(expr.indices, ForAll(free_indices, body),
                          _transpose=transpose)
        else:
            expr = Recipe(expr.indices, body, _transpose=transpose)

        return expr

    def map_let(self, expr, bound_above, bound_below):

        # Indices bound in the Let bindings should not count as
        # bound_below for nodes higher in the tree.
        return Let(tuple((symbol, self.rec(letexpr, bound_above,
                                           bound_below=None))
                         for symbol, letexpr in expr.bindings),
                   self.rec(expr.body, bound_above, bound_below))

    def map_index_sum(self, expr, bound_above, bound_below):
        indices = expr.indices
        for idx in indices:
            bound_above.add(idx)
        body = self.rec(expr.body, bound_above, bound_below)
        for idx in indices:
            bound_above.remove(idx)
        return IndexSum(indices, body)

    def map_for_all(self, expr, bound_above, bound_below):
        indices = expr.indices
        for idx in indices:
            bound_above.add(idx)
        body = self.rec(expr.body, bound_above, bound_below)
        for idx in indices:
            bound_above.remove(idx)
            bound_below.appendleft(idx)
        return ForAll(indices, body)


class IndexSumMapper(IdentityMapper):
    """A mapper that binds unbound IndexSums to temporary variables
    using Lets."""

    def __init__(self, kernel_data):
        """
        :arg context: a mapping from variable names to values
        """
        super(IndexSumMapper, self).__init__()
        self.kernel_data = kernel_data
        self._isum_stack = {}
        self._bound_isums = set()

    def __call__(self, expr):

        if isinstance(expr.body, IndexSum):
            self._bound_isums.add(expr.body)

        return super(IndexSumMapper, self).__call__(expr)

    def _bind_isums(self, expr):
        bindings = []
        if isinstance(expr, Variable):
            children = (expr,)
        elif hasattr(expr, "children"):
            children = expr.children
        else:
            return expr

        for temp in children:
            if temp in self._isum_stack:
                isum = self._isum_stack[temp]
                bindings.append((temp, isum))
        for temp, isum in bindings:
            del self._isum_stack[temp]
        if len(bindings) > 0:
            expr = Let(tuple(bindings), expr)
        return expr

    def map_recipe(self, expr):
        body = self._bind_isums(self.rec(expr.body))
        return Recipe(expr.indices, body)

    def map_let(self, expr):
        # Record IndexSums already bound to a temporary
        new_bindings = []
        for v, e in expr.bindings:
            if isinstance(e, IndexSum):
                self._bound_isums.add(e)
            new_bindings.append((v, self.rec(e)))

        body = self._bind_isums(self.rec(expr.body))
        return Let(tuple(new_bindings), body)

    def map_index_sum(self, expr):
        if expr in self._bound_isums:
            return super(IndexSumMapper, self).map_index_sum(expr)

        # Replace IndexSum with temporary and add to stack
        temp = self.kernel_data.new_variable("isum")
        body = self._bind_isums(self.rec(expr.body))
        expr = IndexSum(expr.indices, body)
        self._isum_stack[temp] = expr
        return temp


class CancelCompoundVectorMapper(IdentityMapper):
    """Mapper to find and expand reductions over CompoundVectors.

    Eventually this probably needs some policy support to decide which
    cases it is worth expanding and cancelling and which not.
    """
    def map_index_sum(self, expr, *args, **kwargs):

        if isinstance(expr.body, (Product, Sum)):
            fs = []
            body = self.rec(expr.body, *args,
                            sum_indices=expr.indices, factored_summands=fs)

            if fs:
                assert len(fs) == 1
                indices = tuple(i for i in expr.indices if i != fs[0][0])
                if indices:
                    return IndexSum(indices, fs[0][1])
                else:
                    return fs[0][1]
            else:
                return IndexSum(expr.indices, body)
        else:
            return super(CancelCompoundVectorMapper, self).map_index_sum(
                expr, *args, **kwargs)

    def map_product(self, expr, sum_indices=None, factored_summands=None, *args,
                    **kwargs):

        try:
            if not sum_indices:
                raise ValueError
            vec_i = None
            vectors = []
            factors = []
            for c in expr.children:
                if isinstance(c, CompoundVector):
                    if vec_i is None:
                        vec_i = c.index
                    elif c.index != vec_i:
                        raise ValueError
                    vectors.append(c)
                else:
                    factors.append(c)

            if vec_i is None:
                raise ValueError  # No CompoundVector

            # Flatten the CompoundVector.
            flattened = 0

            r = {}
            replacer = _IndexMapper(r)
            for i in vec_i.as_range:
                r[vec_i] = i
                prod = 1
                for c in expr.children:
                    prod *= replacer(c)
                flattened += prod

            factored_summands.append((vec_i, flattened))
            return 0.0

        except ValueError:
            # Drop to here if this is not a cancellation opportunity for whatever reason.
            return super(CancelCompoundVectorMapper, self).map_product(
                expr, *args, **kwargs)


class FactorDeltaMapper(IdentityMapper):
    """Mapper to pull deltas up the expression tree to maximise the opportunities for cancellation."""

    def map_product(self, expr, *args, **kwargs):

        children = (self.rec(child, *args, **kwargs) for child in expr.children)
        factors = []
        deltas = []

        for child in children:
            if isinstance(child, Delta):
                deltas.append(child)
                factors.append(child.body)
            else:
                factors.append(child)

        result = flattened_product(tuple(factors))

        for delta in deltas:
            result = Delta(delta.indices, result)

        return result


class CancelDeltaMapper(IdentityMapper):
    """Mapper to cancel and/or replace indices according to the rules for Deltas."""

    # Those nodes through which it is legal to transmit sum_indices.
    _transmitting_nodes = (IndexSum, ForAll, Delta, Let)

    def map_index_sum(self, expr, replace=None, sum_indices=(), *args, **kwargs):

        if replace is None:
            replace = {}

        def flatten(index):
            try:
                return (index,) + reduce((lambda a, b: a + b), map(flatten, index.factors))
            except AttributeError:
                return (index,)

        flattened = map(flatten, expr.indices)
        sum_indices += reduce((lambda a, b: a + b), flattened)

        if type(expr.body) in self._transmitting_nodes:
            # New index replacements are only possible in chains certain ast nodes.
            body = self.rec(expr.body, *args, replace=replace, sum_indices=sum_indices, **kwargs)
        else:
            body = self.rec(expr.body, *args, replace=replace, **kwargs)

        new_indices = []
        for index in flattened:
            if index[0] in replace:
                # Replaced indices are dropped.
                pass
            elif any(i in replace for i in index[1:]):
                for i in index[1:]:
                    if i not in replace:
                        new_indices.append(i)
            else:
                new_indices.append(index[0])

        if new_indices:
            return IndexSum(new_indices, body)
        else:
            return body

    def map_delta(self, expr, replace=None, sum_indices=(), *args, **kwargs):

        # For the moment let's just go with the delta has two indices idea
        assert len(expr.indices) == 2

        if replace is not None:
            indices = tuple(replace[index] if index in replace.keys() else index for index in expr.indices)
        else:
            indices = expr.indices

        if indices[1] in sum_indices:
            replace[indices[1]] = indices[0]
            indices = (indices[0], indices[0])
        elif indices[0] in sum_indices:
            replace[indices[0]] = indices[1]
            indices = (indices[1], indices[1])

        # Only attempt new replacements if we are in transmitting node stacks.
        if sum_indices and indices[0] != indices[1]:
            targets = replace.values()
            if indices[0] in targets and indices[1] not in targets:
                replace[indices[1]] = indices[0]
                indices = (indices[0], indices[0])
            elif indices[1] in targets and indices[0] not in targets:
                replace[indices[0]] = indices[1]
                indices = (indices[0], indices[0])
            # else:
            #    # I don't think this can happen.
            #    raise NotImplementedError

        if type(expr.body) in self._transmitting_nodes:
            # New index replacements are only possible in chains of certain ast nodes.
            body = self.rec(expr.body, *args, replace=replace,
                            sum_indices=sum_indices, **kwargs)
        else:
            body = self.rec(expr.body, *args, replace=replace, **kwargs)

        if indices[0] == indices[1]:
            return body
        else:
            return Delta(indices, body)

    def map_recipe(self, expr, replace=None, *args, **kwargs):
        if replace is None:
            replace = {}

        body = self.rec(expr.body, *args, replace=replace, **kwargs)

        def recurse_replace(index):
            if index in replace:
                return replace[index]
            else:
                try:
                    return type(index)(*map(recurse_replace, index.factors))
                except AttributeError:
                    return index

        if replace:
            indices = tuple(tuple(map(recurse_replace, itype)) for itype in expr.indices)
        else:
            indices = expr.indices

        return Recipe(indices, body)

    def map_let(self, expr, replace=None, sum_indices=(), *args, **kwargs):
        # Propagate changes first into the body. Then do any required
        # substitutions on the bindings.

        body = self.rec(expr.body, *args, replace=replace,
                        sum_indices=sum_indices, **kwargs)

        # Need to think about conveying information from the body to
        # the bindings about diagonalisations which might occur.

        bindings = self.rec(expr.bindings, *args, replace=replace,
                            sum_indices=sum_indices, **kwargs)

        return Let(bindings, body)

    def map_index(self, expr, replace=None, *args, **kwargs):

        return replace[expr] if replace and expr in replace else expr


class _DoNotFactorSet(set):
    """Dummy set object used to indicate that sum factorisation of a subtree is invalid."""
    pass


class SumFactorMapper(IdentityMapper):
    """Mapper to attempt sum factorisation. This is currently a sketch
    implementation which is not safe for particularly general cases."""

    # Internal communication of separable index sets is achieved by
    # the index_groups argument. This is a set containing tuples of
    # grouped indices.

    def __init__(self, kernel_data):

        super(SumFactorMapper, self).__init__()

        self.kernel_data = kernel_data

    @staticmethod
    def factor_indices(indices, index_groups):
        # Determine the factorisability of the expression with the
        # given index_groups assuming an IndexSum over indices.
        if len(indices) != 1 or not isinstance(indices[0], TensorIndex) \
           or len(indices[0].factors) != 2:
            return False

        i = list(indices[0].factors)

        # Try to factor the longest length first.
        if i[0].length < i[1].length:
            i.reverse()

        for n in range(2):
            factorstack = []
            nontrivial = False
            for g in index_groups:
                if i[n] in g:
                    factorstack.append(g)
                elif i[(n+1) % 2] in g:
                    nontrivial = True

            if factorstack and nontrivial:
                return i[n]

        return False

    def map_index_sum(self, expr, index_groups=None, *args, **kwargs):
        """Discover how factorisable this IndexSum is."""

        body_igroups = set()
        body = self.rec(expr.body, *args, index_groups=body_igroups, **kwargs)

        factor_index = self.factor_indices(expr.indices, body_igroups)
        if factor_index:
            factorised = SumFactorSubTreeMapper(factor_index)(expr.body)
            try:
                return factorised.generate_factored_expression(self.kernel_data, expr.indices[0].factors)
            except:
                pass

        return expr.__class__(expr.indices, body)

    def map_index(self, expr, index_groups=None, *args, **kwargs):
        """Add this index into all the sets in index_groups."""

        if index_groups is None:
            return expr
        elif hasattr(expr, "factors"):
            return expr.__class__(*self.rec(expr.factors, *args, index_groups=index_groups, **kwargs))
        elif index_groups:
            news = []
            for s in index_groups:
                news.append(s + (expr,))
            index_groups.clear()
            index_groups.update(news)
        else:
            index_groups.add((expr,))

        return expr

    def map_product(self, expr, index_groups=None, *args, **kwargs):
        """Union of the index groups of the children."""

        if index_groups is None:
            return super(SumFactorMapper, self).map_product(expr, *args, **kwargs)
        else:
            result = 1
            for c in expr.children:
                igroup = set()
                result *= self.rec(c, *args, index_groups=igroup, **kwargs)
                index_groups |= igroup

            return result

    def map_sum(self, expr, index_groups=None, *args, **kwargs):
        """If the summands have the same factors, propagate them up.
        Otherwise (for the moment) put a _DoNotFactorSet in the output.
        """

        igroups = [set() for c in expr.children]
        new_children = [self.rec(c, *args, index_groups=i, **kwargs)
                        for c, i in zip(expr.children, igroups)]

        if index_groups is not None:
            # index_groups really should be empty.
            if index_groups:
                    raise ValueError("Can't happen!")

            # This is not quite safe as it imposes additional ordering.
            if all([i == igroups[0] for i in igroups[1:]]):
                index_groups.update(igroups[0])
            else:
                raise ValueError("Don't know how to do this")

        return flattened_sum(tuple(new_children))


class _Factors(object):
    """A product factorised by the presence or absence of the index provided."""
    def __init__(self, index, expr=None, indices=None):
        self.index = index
        self.factor = 1
        self.remainder = 1

        if expr:
            self.insert(expr, indices)

    def insert(self, expr, indices):
        if self.index in indices:
            self.factor *= expr
        else:
            self.remainder *= expr

    def __imul__(self, other):

        if isinstance(other, _Factors):
            if other.index != self.index:
                raise ValueError("Can only multiply _Factors with the same index")
            self.factor *= other.factor
            self.remainder *= other.remainder
        else:
            self.insert(*other)
        return self

    def generate_factored_expression(self, kernel_data, indices):
        # Generate the factored expression using the set of indices provided.
        indices = list(indices)
        indices.remove(self.index)

        temp = kernel_data.new_variable("isum")
        d = tuple(i for i in indices if isinstance(i, DimensionIndex))
        b = tuple(i for i in indices if isinstance(i, BasisFunctionIndexBase))
        p = tuple(i for i in indices if isinstance(i, PointIndexBase))
        return Let(((temp, Recipe((d, b, p), IndexSum((self.index,), self.factor))),),
                   IndexSum(tuple(indices), temp[d+b+p]*self.remainder))


class _FactorSum(object):
    """A sum of _Factors."""

    def __init__(self, factors=None):

        self.factors = list(factors or [])

    def __iadd__(self, factor):

        self.factors.append(factor)
        return self

    def __imul__(self, other):

        assert isinstance(other, _Factors)
        for f in self.factors:
            f *= other
        return self

    def generate_factored_expression(self, kernel_data, indices):
        # Generate the factored expression using the set of indices provided.

        genexprs = [f.generate_factored_expression(kernel_data, indices)
                    for f in self.factors]

        # This merges some indexsums but that gets in the way of delta cancellation.
        if all(self.factors[0].index == f.index for f in self.factors[1:]):
            return Let(tuple(g.bindings[0] for g in genexprs),
                        flattened_sum(tuple(g.body for g in genexprs)))
        else:
            return flattened_sum(genexprs)


class SumFactorSubTreeMapper(IdentityMapper):
    """Mapper to actually impose a defined factorisation on a subtree."""

    def __init__(self, factor_index):

        super(SumFactorSubTreeMapper, self).__init__()

        # The index with respect to which the factorisation should occur.
        self.factor_index = factor_index

    def map_index(self, expr, indices=None, *args, **kwargs):
        """Add this index into all the sets in index_groups."""

        if indices is None:
            return expr
        elif hasattr(expr, "factors"):
            return expr.__class__(*self.rec(expr.factors, *args, indices=indices, **kwargs))
        else:
            indices.add(expr)

        return expr

    def map_product(self, expr, indices=None, *args, **kwargs):

        f = _Factors(self.factor_index)
        for c in expr.children:
            i = set()
            rc = self.rec(c, *args, indices=i, **kwargs)
            if isinstance(rc, _FactorSum):
                rc *= f
                f = rc
            elif isinstance(rc, _Factors):
                f *= rc
            else:
                f *= (rc, i)

        return f

    def map_sum(self, expr, indices=None, *args, **kwargs):

        f = _FactorSum()
        for c in expr.children:
            i = set()
            rc = self.rec(c, *args, indices=i, **kwargs)
            if isinstance(rc, (_Factors, _FactorSum)):
                f += rc
            else:
                f += _Factors(self.factor_index, rc, i)

        return f
