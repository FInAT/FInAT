from __future__ import absolute_import, print_function, division
from six import iteritems, itervalues
from six.moves import range

import numpy as np

from FIAT.polynomial_set import mis
from FIAT.reference_element import (ufc_simplex, POINT, LINE,
                                    TRIANGLE, QUADRILATERAL,
                                    TETRAHEDRON, TENSORPRODUCT)

from finat.fiat_elements import Lagrange, DiscontinuousLagrange
from finat.finiteelementbase import FiniteElementBase
from finat.tensor_product import TensorProductElement


class TraceError(Exception):
    '''Exception raised when attempting to perform a mathematically
    illegal operation on a trace element, such as gradient evaluations
    or evaluating basis functions in cell interiors.
    '''

    def __init__(self, msg):
        super(TraceError, self).__init__(msg)
        self.msg = msg


class TraceFiniteElement(FiniteElementBase):
    '''
    '''

    def __init__(self, cell, degree, family):
        '''
        '''

        super(TraceFiniteElement, self).__init__()

        sd = cell.get_spatial_dimension()
        if sd in (0, 1):
            raise NotImplementedError("Traces on %d-dim cell not implemented." % sd)

        if cell.get_shape() == TENSORPRODUCT:
            try:
                degree = tuple(degree)
            except TypeError:
                degree = (degree) * len(cell.cells)
            assert len(degree) == len(cell.cells), (
                "Number degrees must be equal to the number of cells."
            )
            if family == 'H1':
                assert min(degree) >= 1, (
                    "H1 Trace on a 0-order element doesn't make sense."
                )
        else:
            if cell.get_shape() not in [TRIANGLE, QUADRILATERAL, TETRAHEDRON]:
                raise NotImplementedError("Trace on a %s not implemented." % type(cell))

            # Cannot have varying degrees for cells of this type
            if isinstance(degree, tuple):
                raise ValueError("Must have a tensor product cell if providing multiple degrees.")

            if family == 'H1':
                assert degree >= 1, (
                    "H1 Trace on a 0-order element doesn't make sense."
                )

        facet_sd = sd - 1
        facet_elements = {}
        topology = cell.get_topology()
        for t_dim, entities in iteritems(topology):
            c = cell.construct_subelement(top_dim)

            if c.get_spatial_dimension() == facet_sd:
                facet_elements[t_dim] = facet_element(family, cell, degree)

        self.cell = cell
        self.degree = degree
        self._facet_elements = facet_elements

    @property
    def cell(self):
        return self.cell

    @property
    def degree(self):
        return self.degree

    @property
    def formdegree(self):
        return self.cell.get_spatial_dimension() - 1

    def entity_dofs(self):
        return self._entity_dofs

    def space_dimension(self):
        return self._space_dimension

    @property
    def index_shape(self):
        return (self.space_dimension(),)

    @property
    def value_shape(self):
        return ()

    def basis_evaluation(self, order, ps, entity):
        '''
        '''

        assert entity is not None, "Must specify an entity to tabulate on."

        sd = self.cell.get_spatial_dimension()
        facet_sd = sd - 1

        phivals = {}
        for i in range(order + 1):
            alphas = mis(sd, i)
            for alpha in alphas:
                phivals[alpha] = np.zeros(shape=(self.space_dimension(), len(ps)))

        eval_key = (0,) * sd

        entity_dim, _ = entity

        if entity_dim not in self.facet_elements:
            msg = "The trace element can only be tabulated on facets."
            for key in phivals:
                phivals[key] = TraceError(msg)

            return phivals

        else:
            offset = 0
            for facet_dim in sorted(self.facet_elements):
                element = self.facet_elements[facet_dim]
                nf = element.space_dimension()
                num_facets = len(self.cell.get_topology()[facet_dim])

                for i in range(num_facets):
                    if (facet_dim, i) == entity:
                        nonzerovals, = itervalues(element.basis_evaluation(0, ps))
                        indices = slice(offset, offset + nf)

                    offset += nf

        if order > 0:
            msg = "Gradient evaluations on trace elements are not well-defined."
            for key in phivals:
                if key != eval_key:
                    phivals[key] = TraceError(msg)

        phivals[eval_key][indices, :] = nonzerovals

        return phivals

    def point_evaluation(self, order, refcoords, entity=None):
        return NotImplementedError

    @property
    def mapping(self):
        return "affine"


def facet_element(family, cell, degree):
    '''Constructs the relevant facet elements for the trace element. If the
    trace is 'H1', then continuous Lagrange elements are returned. A
    discontinuous Lagrange element is provided for the 'HDiv' trace.

    :arg family: A string denoting the family of the trace element. Supported
                 trace families include: 'H1' and 'HDiv'.
    :arg cell: A facet cell to define the element on.
    :arg degree: The degree of the facet element.
    '''

    if family not in ('H1', 'HDiv'):
        raise NotImplementedError("Trace of an %s-element is not implemented." % family)

    facet_elements = {'H1': Lagrange,
                      'HDiv': DiscontinuousLagrange}

    if cell.get_shape() in [LINE, TRIANGLE]:
        element = facet_elements[family](cell, degree)

    elif cell.get_shape() == QUADRILATERAL:
        A = facet_elements[family](ufc_simplex(1), degree)
        B = facet_elements[family](ufc_simplex(1), degree)
        element = TensorProductElement(A, B)

    elif cell.get_shape() == TENSORPRODUCT:
        assert len(degree) == len(cell.cells), (
            "Number degrees must be equal to the number of cells."
        )
        sub_elements = [facet_element(family, c, d)
                        for c, d in zip(cell.cells, degree)
                        if c.get_shape() != POINT]

        if len(sub_elements) > 1:
            element = TensorProductElement(*sub_elements)
        else:
            element, = sub_elements

    else:
        raise NotImplementedError("%s cells are not implemented." type(cell))

    return element
