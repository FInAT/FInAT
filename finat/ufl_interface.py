"""Provide interface functions which take UFL objects and return FInAT ones."""
import finat
import FIAT


def _q_element(cell, degree):
    # Produce a Q element from GLL elements.

    if cell.get_spatial_dimension() == 1:
        return finat.GaussLobatto(cell, degree)
    else:
        return finat.ScalarProductElement(_q_element(cell.A, degree),
                                          _q_element(cell.B, degree))


_cell_map = {
    "triangle": FIAT.reference_element.UFCTriangle(),
    "interval": FIAT.reference_element.UFCInterval(),
    "quadrilateral": FIAT.reference_element.FiredrakeQuadrilateral()
}

_element_map = {
    "Lagrange": finat.Lagrange,
    "Discontinuous Lagrange": finat.DiscontinuousLagrange,
    "Q": _q_element
}


def cell_from_ufl(cell):

    return _cell_map[cell.cellname()]


def element_from_ufl(element):

    # Need to handle the product cases.

    return _element_map[element.family()](cell_from_ufl(element.cell()),
                                          element.degree())
