try:
    from pyop2.pyparloop import Kernel
except:
    Kernel = None
from .interpreter import evaluate


def pyop2_kernel(kernel, kernel_args, interpreter=False):
    """Return a :class:`pyop2.Kernel` from the recipe and kernel data
    provided.

    :param kernel: The :class:`~.utils.Kernel` to map to PyOP2.
    :param kernel_args: The ordered list of Pymbolic variables constituting
      the kernel arguments, excluding the result of the recipe (the latter
      should be prepended to the argument list).
    :param interpreter: If set to ``True``, the kernel will be
      evaluated using the FInAT interpreter instead of generating a
      compiled kernel.

    :result: The :class:`pyop2.Kernel`
    """

    if Kernel is None:
        raise ImportError("pyop2 was not imported. Is it installed?")

    if kernel_args and \
       set(kernel_args) != kernel.kernel_data.kernel_args:
        raise ValueError("Incomplete value list")

    if interpreter:

        def kernel_function(*args):
            context = {kernel_args: args[1:]}

            args[0][:] = evaluate(kernel.recipe, context, kernel.kernel_data)

        return Kernel(kernel_function)

    else:
        raise NotImplementedError
