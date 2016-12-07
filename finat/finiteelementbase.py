from __future__ import absolute_import, print_function, division
from six import with_metaclass

from abc import ABCMeta, abstractproperty, abstractmethod

import gem


class FiniteElementBase(with_metaclass(ABCMeta)):

    @property
    def cell(self):
        '''The reference cell on which the element is defined.
        '''

        return self._cell

    @property
    def degree(self):
        '''The degree of the embedding polynomial space.

        In the tensor case this is a tuple.
        '''

        return self._degree

    @abstractproperty
    def index_shape(self):
        '''A tuple indicating the number of degrees of freedom in the
        element. For example a scalar quadratic Lagrange element on a triangle
        would return (6,) while a vector valued version of the same element
        would return (6, 2)'''

    @abstractproperty
    def value_shape(self):
        '''A tuple indicating the shape of the element.'''

    def get_indices(self):
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        return tuple(gem.Index(extent=d) for d in self.index_shape)

    def get_value_indices(self):
        '''A tuple of GEM :class:`~gem.Index` of the correct extents to loop over
        the value shape of this element.'''

        return tuple(gem.Index(extent=d) for d in self.value_shape)

    @abstractmethod
    def basis_evaluation(self, order, ps, entity=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param order: return derivatives up to this order.
        :param ps: the point set object.
        :param entity: the cell entity on which to tabulate.
        '''

    @property
    def preferred_quadrature(self):
        '''A list of quadrature rules whose structure this element is capable
        of exploiting. Each entry in the list should be a pair (rule,
        degree), where the degree might be `None` if the element has
        no preferred quadrature degree.'''

        return ()

    def __hash__(self):
        """Elements are equal if they have the same class, degree, and cell."""

        return hash((type(self), self._cell, self._degree))

    def __eq__(self, other):
        """Elements are equal if they have the same class, degree, and cell."""

        return type(self) == type(other) and self._cell == other._cell and\
            self._degree == other._degree
