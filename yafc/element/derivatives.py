
class Derivative(object):
    "Abstract symbolic object for a derivative."

    def __init__(self, name, doc):
        self.__doc__ = doc
        self.name = name

    def __str__(self):
        return self.name

grad = Derivative("grad", "Symbol for the gradient operation")
div = Derivative("div", "Symbol for the divergence operation")
curl = Derivative("curl", "Symbol for the curl operation")
