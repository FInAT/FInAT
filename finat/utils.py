import pymbolic.primitives as p
import inspect
from functools import wraps
import FIAT

"""
from http://stackoverflow.com/questions/2025562/inherit-docstrings-in-python-class-inheritance

doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
"""


class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit


class KernelData(object):
    def __init__(self, coordinate_element, affine=None):
        """
        :param coordinate_element: the (vector-valued) finite element for
            the coordinate field.
        :param affine: Specifies whether the pullback is affine (and therefore
            whether the Jacobian must be evaluated at every quadrature point).
            If not specified, this is inferred from coordinate_element.
        """

        self.coordinate_element = coordinate_element
        if affine is None:
            self.affine = coordinate_element.degree <= 1 \
                and isinstance(coordinate_element.cell, _simplex)
        else:
            self.affine = True

        self.static = {}
        self.params = {}
        self.geometry = {}

    @property
    def J(self):

        try:
            return self.geometry["J"]
        except KeyError:
            self.geometry["J"] = p.Variable("J")
            return self.geometry["J"]

    @property
    def invJ(self):

        # ensure J exists
        self.J

        try:
            return self.geometry["invJ"]
        except KeyError:
            self.geometry["invJ"] = p.Variable("invJ")
            return self.geometry["invJ"]

    @property
    def detJ(self):

        # ensure J exists
        self.J

        try:
            return self.geometry["detJ"]
        except KeyError:
            self.geometry["detJ"] = p.Variable("detJ")
            return self.geometry["detJ"]

# Tuple of simplex cells. This relies on the fact that FIAT only
# defines simplex elements.
_simplex = tuple(e for e in FIAT.reference_element.__dict__.values()
                 if (inspect.isclass(e)
                     and issubclass(e, FIAT.reference_element.ReferenceElement)
                     and e is not FIAT.reference_element.ReferenceElement))
