"""A testing utility that compiles and executes COFFEE ASTs to
evaluate a given recipe. Provides the same interface as FInAT's
internal interpreter. """

from pymbolic.mapper import CombineMapper
from greek_alphabet import translate_symbol
import coffee.base as coffee
import os
import subprocess
import ctypes
import numpy as np
from .utils import Kernel
from .ast import Recipe, IndexSum, Array, Inverse
from .mappers import BindingMapper, IndexSumMapper
from pprint import pformat
from collections import deque


determinant = {1: lambda e: coffee.Determinant1x1(e),
               2: lambda e: coffee.Determinant2x2(e),
               3: lambda e: coffee.Determinant3x3(e)}


class CoffeeMapper(CombineMapper):
    """A mapper that generates Coffee ASTs for FInAT expressions"""

    def __init__(self, kernel_data, varname="A", increment=False):
        """
        :arg context: a mapping from variable names to values
        :arg varname: name of the implied outer variable
        :arg increment: flag indicating that the kernel should
             increment result values instead of assigning them
        """
        super(CoffeeMapper, self).__init__()
        self.kernel_data = kernel_data
        self.scope_var = deque()
        self.scope_ast = deque()
        if increment:
            self.scope_var.append((varname, coffee.Incr))
        else:
            self.scope_var.append((varname, coffee.Assign))

    def _push_scope(self):
        self.scope_ast.append([])

    def _pop_scope(self):
        return self.scope_ast.pop()

    def _create_loop(self, index, body):
        itvar = self.rec(index)
        extent = index.extent
        init = coffee.Decl("int", itvar, extent.start or 0)
        cond = coffee.Less(itvar, extent.stop)
        incr = coffee.Incr(itvar, extent.step or 1)
        return coffee.For(init, cond, incr, coffee.Block(body, open_scope=True))

    def combine(self, values):
        return list(values)

    def map_recipe(self, expr):
        return self.rec(expr.body)

    def map_variable(self, expr):
        return coffee.Symbol(translate_symbol(expr.name))

    map_index = map_variable

    def map_constant(self, expr):
        return expr.real

    def map_subscript(self, expr):
        name = translate_symbol(expr.aggregate.name)
        if isinstance(expr.index, tuple):
            indices = self.rec(expr.index)
        else:
            indices = (self.rec(expr.index),)
        return coffee.Symbol(name, rank=indices)

    def map_index_sum(self, expr):
        return self.rec(expr.body)

    def map_product(self, expr):
        prod = self.rec(expr.children[0])
        for factor in expr.children[1:]:
            prod = coffee.Prod(prod, self.rec(factor))
        return prod

    def map_inverse(self, expr):
        e = expr.expression
        return coffee.Invert(self.rec(e), e.shape[0])

    def map_det(self, expr):
        e = expr.expression
        return determinant[e.shape[0]](self.rec(e))

    def map_abs(self, expr):
        return coffee.FunCall("fabs", self.rec(expr.expression))

    def map_for_all(self, expr):
        name, stmt = self.scope_var[-1]
        var = coffee.Symbol(name, self.rec(expr.indices))
        self._push_scope()
        body = self.rec(expr.body)
        scope = self._pop_scope()
        body = scope + [stmt(var, body)]
        for idx in expr.indices:
            body = [self._create_loop(idx, body)]
        return coffee.Block(body)

    def map_let(self, expr):
        for v, e in expr.bindings:
            shape = v.shape if isinstance(v, Array) else ()
            var = coffee.Symbol(self.rec(v), rank=shape)
            self.scope_var.append((v, coffee.Assign))

            self._push_scope()
            body = self.rec(e)
            scope = self._pop_scope()

            if isinstance(e, IndexSum):
                # Recurse on expression in a new scope
                lbody = scope + [coffee.Incr(var, body)]

                # Construct IndexSum loop and add to current scope
                self.scope_ast[-1].append(coffee.Decl("double", var, init="0."))
                self.scope_ast[-1].append(self._create_loop(e.indices[0], lbody))

            elif isinstance(e, Inverse):
                # Coffee currently inverts matrices in-place
                # so we need to memcpy the source matrix first
                mcpy = coffee.FlatBlock("memcpy(%s, %s, %d*sizeof(double));\n" %
                                        (v, e.expression, shape[0] * shape[1]))
                e.expression = v
                self.scope_ast[-1].append(coffee.Decl("double", var))
                self.scope_ast[-1].append(mcpy)
                self.scope_ast[-1].append(self.rec(e))

            elif isinstance(body, coffee.Expr):
                self.scope_ast[-1].append(coffee.Decl("double", var, init=body))
            else:
                self.scope_ast[-1].append(coffee.Decl("double", var))
                self.scope_ast[-1].append(body)

            self.scope_var.pop()
        return self.rec(expr.body)


class CoffeeKernel(Kernel):

    def __init__(self, recipe, kernel_data):
        super(CoffeeKernel, self).__init__(recipe, kernel_data)

        # Apply mapper to bind all IndexSums to temporaries
        self.recipe = IndexSumMapper(self.kernel_data)(self.recipe)

        # Apply pre-processing mapper to bind free indices
        self.recipe = BindingMapper(self.kernel_data)(self.recipe)

    def generate_ast(self, kernel_args=None, varname="A", increment=False):
        if kernel_args is None:
            kernel_args = self.kernel_data.kernel_args
        args_ast = []
        body_ast = []

        mapper = CoffeeMapper(self.kernel_data, varname=varname,
                              increment=increment)

        # Add argument declarations
        for var in kernel_args:
            if isinstance(var, Array):
                var_ast = coffee.Symbol(var.name, var.shape)
            else:
                var_ast = coffee.Symbol("**" + var.name)
            args_ast.append(coffee.Decl("double", var_ast))

        # Write AST to initialise static kernel data
        for data in self.kernel_data.static.values():
            values = data[1]()
            val_str = pformat(values.tolist())
            val_str = val_str.replace('[', '{').replace(']', '}')
            val_init = coffee.ArrayInit(val_str)
            var = coffee.Symbol(mapper(data[0]), values.shape)
            body_ast.append(coffee.Decl("double", var, init=val_init))

        # Convert the kernel recipe into an AST
        body_ast.append(mapper(self.recipe))

        return coffee.FunDecl("void", "finat_kernel", args_ast,
                              coffee.Block(body_ast),
                              headers=["math.h", "string.h"])


def evaluate(expression, context={}, kernel_data=None):
    index_shape = ()
    args_data = []

    # Pack free indices as kernel arguments
    for index in expression.indices:
        for i in index:
            index_shape += (i.extent.stop, )
    index_data = np.empty(index_shape, dtype=np.double)
    args_data.append(index_data.ctypes.data)
    kernel_data.kernel_args = [Array("A", shape=index_shape)]

    # Pack context arguments
    for var, value in context.iteritems():
        kernel_data.kernel_args.append(Array(var, shape=value.shape))
        args_data.append(value.ctypes.data)

    # Generate kernel function
    kernel = CoffeeKernel(expression, kernel_data).generate_ast()

    basename = os.path.join(os.getcwd(), "finat_kernel")
    with file(basename + ".c", "w") as f:
        f.write(str(kernel))

    # Compile kernel function into .so
    cc = [os.environ['CC'] if "CC" in os.environ else 'gcc']
    cc += ['-Wall', '-O0', '-g', '-fPIC', '-shared', '-std=c99']
    cc += ["-o", "%s.so" % basename, "%s.c" % basename]
    cc += ['-llapack', '-lblas']
    try:
        subprocess.check_call(cc)
    except subprocess.CalledProcessError as e:
        print "Compilation error: ", e
        raise Exception("Failed to compile %s.c" % basename)

    # Load compiled .so
    try:
        kernel_lib = ctypes.cdll.LoadLibrary(basename + ".so")
    except OSError as e:
        print "Library load error: ", e
        raise Exception("Failed to load %s.so" % basename)

    # Invoke compiled kernel with packed arguments
    kernel_lib.finat_kernel(*args_data)

    # Close compiled kernel library
    ctypes.cdll.LoadLibrary('libdl.so').dlclose(kernel_lib._handle)

    return index_data
