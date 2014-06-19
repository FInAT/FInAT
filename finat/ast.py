class Recipe(object):
    """AST snippets and data corresponding to some form of finite element
    evaluation."""
    def __init__(self, indices, instructions, depends):
        self._indices = indices
        self._instructions = instructions
        self._depends = depends

    @property
    def indices(self):
        '''The free indices in this :class:`Recipe`.'''

        return self._indices

    @property
    def instructions(self):
        '''The actual instructions making up this :class:`Recipe`.'''

        return self._instructions

    @property
    def depends(self):
        '''The input fields of this :class:`Recipe`.'''

        return self._depends
