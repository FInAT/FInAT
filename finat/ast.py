from pymbolic.mapper import IdentityMapper


class _IndexMapper(IdentityMapper):
    def __init__(self, replacements):
        super(_IndexMapper, self).__init__()

        self.replacements = replacements

    def map_index(self, expr):
        '''Replace indices if they are in the replacements list'''
        try:
            return(self.replacements[expr])
        except IndexError:
            return expr

    def map_foreign(self, expr, *args):

        if isinstance(expr, Recipe):
            return Recipe.replace_indices(self.replacements)
        else:
            return super(_IndexMapper, self).map_foreign(expr, *args)


# Probably Recipe should be a pymbolic.Expression
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

    def __getitem__(self, index):

        replacements = {}

        try:
            for i in range(len(index)):
                if index[i] == slice(None):
                    # Don't touch colon indices.
                    pass
                else:
                    replacements[self.indices[i]] = index[i]
        except TypeError:
            # Index wasn't iterable.
            replacements[self.indices[0]] = index

        return self.replace_indices(replacements)

    def replace_indices(self, replacements):
        """Return a copy of this :class:`Recipe` with some of the indices
        substituted."""

        return _IndexMapper(replacements)(self)
