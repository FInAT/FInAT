"""A testing utility that compiles and executes COFFEE ASTs to
evaluate a given recipe. Provides the same interface as FInAT's
internal interpreter. """

def evaluate(expression, context={}, kernel_data=None):
    print expression
