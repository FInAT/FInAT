from __future__ import absolute_import, print_function, division
from six import iteritems

import numpy

import FIAT

import gem
from gem.interpreter import evaluate

from finat.point_set import PointSet


class FinatElementWrapper(FIAT.FiniteElement):

    def __init__(self, finat_element):
        # FInAT elements have no dual API at the moment,
        # so not calling __init__ of the superclass.
        self._element = finat_element

    def get_reference_element(self):
        """Return the reference element for the finite element."""
        return self._element.cell

    def get_dual_set(self):
        """Return the dual for the finite element."""
        raise NotImplementedError("Sadly, FInAT cannot provide the dual set!")

    def get_order(self):
        """Return the order of the element (may be different from the degree)."""
        return self._element.degree

    def dual_basis(self):
        """Return the dual basis (list of functionals) for the finite
        element."""
        raise NotImplementedError("Sadly, FInAT cannot provide the dual basis!")

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self._element.entity_dofs()

    def entity_closure_dofs(self):
        """Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element."""
        return self._element.entity_closure_dofs()

    def get_formdegree(self):
        """Return the degree of the associated form (FEEC)"""
        return self._element.formdegree

    def mapping(self):
        """Return a list of appropriate mappings from the reference
        element to a physical element for each basis function of the
        finite element."""
        return [self._element.mapping] * self.space_dimension()

    def num_sub_elements(self):
        """Return the number of sub-elements."""
        return 1

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return self._element.space_dimension()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        beta = self._element.get_indices()
        zeta = self._element.get_value_indices()

        ps = PointSet(points)
        # We use this so the point index will not be missing
        one = gem.Indexed(gem.Literal(numpy.ones(len(ps.points))), ps.indices)

        finat_result = self._element.basis_evaluation(order, ps, entity)
        result = {}
        for alpha, finat_table in iteritems(finat_result):
            # Convert GEM Failure nodes back to Exception instances
            if isinstance(finat_table, gem.Failure):
                result[alpha] = finat_table.exception
                continue

            table_roll = gem.ComponentTensor(
                gem.Product(one, gem.Indexed(finat_table, beta + zeta)),
                beta + zeta + ps.indices
            )
            table_result, = evaluate([table_roll])
            assert not table_result.fids
            table_array = table_result.arr

            shape = (self.space_dimension(),) + self.value_shape() + (len(ps.points),)
            result[alpha] = numpy.reshape(table_array, shape)
        return result

    @staticmethod
    def is_nodal():
        """True if primal and dual bases are orthogonal. If false,
        dual basis is not implemented or is undefined.

        Subclasses may not necessarily be nodal, unless it is a CiarletElement.
        """
        return False

    def degree(self):
        "Return the degree of the (embedding) polynomial space."
        return self._element.degree

    def value_shape(self):
        "Return the value shape of the finite element functions."
        return self._element.value_shape
