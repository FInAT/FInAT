try:
    from pyop2 import Kernel
except:
    Kernel = None
from .interpreter import evaluate
from .ast import Array
from .coffee_compiler import CoffeeKernel


def pyop2_kernel_function(kernel, kernel_args, interpreter=False):
    """Return a python function suitable to be called from PyOP2 from the
    recipe and kernel data provided.

    :param kernel: The :class:`~.utils.Kernel` to map to PyOP2.
    :param kernel_args: The ordered list of Pymbolic variables constituting
      the kernel arguments, excluding the result of the recipe (the latter
      should be prepended to the argument list).
    :param interpreter: If set to ``True``, the kernel will be
      evaluated using the FInAT interpreter instead of generating a
      compiled kernel.

    :result: A function which will execute the kernel.

    """

    if Kernel is None:
        raise ImportError("pyop2 was not imported. Is it installed?")

    if kernel_args and \
       set(kernel_args) != kernel.kernel_data.kernel_args:
        raise ValueError("Incomplete value list")

    if interpreter:

        def kernel_function(*args):
            context = {var.name: val for (var, val) in zip(kernel_args, args[1:])}

            args[0][:] += evaluate(kernel.recipe, context, kernel.kernel_data)

        return kernel_function

    else:
        index_shape = ()
        for index in kernel.recipe.indices:
            for i in index:
                index_shape += (i.extent.stop, )
        kernel_args.insert(0, Array("*A", shape=index_shape))

        coffee_kernel = CoffeeKernel(kernel.recipe, kernel.kernel_data)
        kernel_ast = coffee_kernel.generate_ast(kernel_args=kernel_args,
                                                varname="*A", increment=True)
        return Kernel(kernel_ast, "finat_kernel")
