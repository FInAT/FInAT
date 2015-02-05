"""A testing utility that compiles and executes COFFEE ASTs to
evaluate a given recipe. Provides the same interface as FInAT's
internal interpreter. """

import coffee.base as coffee
import os
import subprocess
import ctypes
import numpy as np
from .utils import Kernel


class CoffeeKernel(Kernel):

    def generate_ast(self, context):
        kernel_args = self.kernel_data.kernel_args
        args_ast = []

        # Generate declaration of result argument
        result_shape = ()
        for index in self.recipe.indices:
            for i in index:
                result_shape += (i.extent.stop,)
        result_ast = coffee.Symbol(kernel_args[0], result_shape)
        args_ast.append(coffee.Decl("double", result_ast))

        # Add argument declarations
        for var in kernel_args[1:]:
            var_ast = coffee.Symbol(str(var), context[var].shape)
            args_ast.append(coffee.Decl("double", var_ast))

        body = coffee.EmptyStatement(None, None)
        return coffee.FunDecl("void", "coffee_kernel", args_ast,
                              body, headers=["stdio.h"])


def evaluate(expression, context={}, kernel_data=None):
    index_shape = ()
    args_data = []

    # Pack free indices as kernel arguments
    for index in expression.indices:
        for i in index:
            index_shape += (i.extent.stop, )
    index_data = np.empty(index_shape, dtype=np.double)
    args_data.append(index_data.ctypes.data)
    kernel_data.kernel_args = ["A"]

    # Pack context arguments
    for var, value in context.iteritems():
        kernel_data.kernel_args.append(var)
        args_data.append(value.ctypes.data)

    # Generate kernel function
    kernel = CoffeeKernel(expression, kernel_data).generate_ast(context)
    basename = os.path.join(os.getcwd(), "coffee_kernel")
    with file(basename + ".c", "w") as f:
        f.write(str(kernel))

    # Compile kernel function into .so
    cc = [os.environ['CC'] if "CC" in os.environ else 'gcc']
    cc += ['-Wall', '-O0', '-g', '-fPIC', '-shared', '-std=c99']
    cc += ["-o", "%s.so" % basename, "%s.c" % basename]
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
    kernel_lib.coffee_kernel(*args_data)

    # Close compiled kernel library
    ctypes.cdll.LoadLibrary('libdl.so').dlclose(kernel_lib._handle)

    return index_data
