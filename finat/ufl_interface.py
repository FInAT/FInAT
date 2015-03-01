"""Provide interface functions which take UFL objects and return FInAT ones."""
import finat
import FIAT

_cell_map = {
    "triangle": FIAT.reference_element.UFCTriangle(),
    "interval": FIAT.reference_element.UFCInterval()
}

_element_map = {
    "Lagrange": finat.Lagrange,
    "Discontinuous Lagrange": finat.DiscontinuousLagrange
}


def cell_from_ufl(cell):

    return _cell_map[cell.cellname()]


def element_from_ufl(element):

    # Need to handle the product cases.

    return _element_map[element.family()](cell_from_ufl(element.cell()),
                                          element.degree())
