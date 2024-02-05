"""This module defines the UFL finite element classes."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file was originally part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Anders Logg 2014
# Modified by Massimiliano Leoni, 2016
# Modified by Matthew Scroggs, 2023

from ufl.cell import TensorProductCell, as_cell
from finat.ufl.elementlist import canonical_element_description, simplices
from finat.ufl.finiteelementbase import FiniteElementBase
from ufl.utils.formatting import istr


class FiniteElement(FiniteElementBase):
    """The basic finite element class for all simple finite elements."""
    # TODO: Move these to base?
    __slots__ = ("_short_name", "_sobolev_space",
                 "_mapping", "_variant", "_repr")

    def __new__(cls,
                family,
                cell=None,
                degree=None,
                form_degree=None,
                quad_scheme=None,
                variant=None):
        """Intercepts construction to expand CG, DG, RTCE and RTCF spaces on TensorProductCells."""
        if cell is not None:
            cell = as_cell(cell)

        if isinstance(cell, TensorProductCell):
            # Delay import to avoid circular dependency at module load time
            from finat.ufl.enrichedelement import EnrichedElement
            from finat.ufl.hdivcurl import HCurlElement as HCurl
            from finat.ufl.hdivcurl import HDivElement as HDiv
            from finat.ufl.tensorproductelement import TensorProductElement

            family, short_name, degree, value_shape, reference_value_shape, sobolev_space, mapping = \
                canonical_element_description(family, cell, degree, form_degree)

            if family in ["RTCF", "RTCE"]:
                cell_h, cell_v = cell.sub_cells()
                if cell_h.cellname() != "interval":
                    raise ValueError(f"{family} is available on TensorProductCell(interval, interval) only.")
                if cell_v.cellname() != "interval":
                    raise ValueError(f"{family} is available on TensorProductCell(interval, interval) only.")

                C_elt = FiniteElement("CG", "interval", degree, variant=variant)
                D_elt = FiniteElement("DG", "interval", degree - 1, variant=variant)

                CxD_elt = TensorProductElement(C_elt, D_elt, cell=cell)
                DxC_elt = TensorProductElement(D_elt, C_elt, cell=cell)

                if family == "RTCF":
                    return EnrichedElement(HDiv(CxD_elt), HDiv(DxC_elt))
                if family == "RTCE":
                    return EnrichedElement(HCurl(CxD_elt), HCurl(DxC_elt))

            elif family in ["NCF", "NCE"]:
                sub_cells = cell.sub_cells()
                if len(sub_cells) == 3:
                    if any(cell.cellname() != "interval" for cell in sub_cells):
                        raise ValueError("%s is available on TensorProductCell(interval, interval, interval) only." % family)
                    cell_h = TensorProductCell(*sub_cells[:-1])
                    cell_v = sub_cells[-1]
                elif len(sub_cells) == 2:
                    cell_h, cell_v = sub_cells
                    if cell_h.cellname() != "quadrilateral":
                        raise ValueError("%s is available on TensorProductCell(quadrilateral, interval) only." % family)
                    if cell_v.cellname() != "interval":
                        raise ValueError("%s is available on TensorProductCell(quadrilateral, interval) only." % family)
                else:
                    raise ValueError("%s is available on TensorProductCell(quadrilateral, interval) only." % family)

                Ic_elt = FiniteElement("CG", cell_v, degree, variant=variant)
                Id_elt = FiniteElement("DG", cell_v, degree - 1, variant=variant)
                if family == "NCF":
                    Qc_elt = FiniteElement("RTCF", cell_h, degree, variant=variant)
                    Qd_elt = FiniteElement("DQ", cell_h, degree - 1, variant=variant)
                else:
                    Qc_elt = FiniteElement("Q", cell_h, degree, variant=variant)
                    Qd_elt = FiniteElement("RTCE", cell_h, degree, variant=variant)

                components = [(Qc_elt, Id_elt), (Qd_elt, Ic_elt)]
                wrapper = HDiv if family == "NCF" else HCurl
                return EnrichedElement(*[wrapper(TensorProductElement(*factors, cell=cell)) for factors in components])

            elif family == "Q":
                return TensorProductElement(*[FiniteElement("CG", c, degree, variant=variant)
                                              for c in cell.sub_cells()],
                                            cell=cell)

            elif family == "DQ":
                def dq_family(cell):
                    """Doc."""
                    return "DG" if cell.cellname() in simplices else "DQ"
                return TensorProductElement(*[FiniteElement(dq_family(c), c, degree, variant=variant)
                                              for c in cell.sub_cells()],
                                            cell=cell)

            elif family == "DQ L2":
                def dq_family_l2(cell):
                    """Doc."""
                    return "DG L2" if cell.cellname() in simplices else "DQ L2"
                return TensorProductElement(*[FiniteElement(dq_family_l2(c), c, degree, variant=variant)
                                              for c in cell.sub_cells()],
                                            cell=cell)

        return super(FiniteElement, cls).__new__(cls)

    def __init__(self,
                 family,
                 cell=None,
                 degree=None,
                 form_degree=None,
                 quad_scheme=None,
                 variant=None):
        """Create finite element.

        Args:
            family: The finite element family
            cell: The geometric cell
            degree: The polynomial degree (optional)
            form_degree: The form degree (FEEC notation, used when field is
               viewed as k-form)
            quad_scheme: The quadrature scheme (optional)
            variant: Hint for the local basis function variant (optional)
        """
        # Note: Unfortunately, dolfin sometimes passes None for
        # cell. Until this is fixed, allow it:
        if cell is not None:
            cell = as_cell(cell)

        (
            family, short_name, degree, value_shape, reference_value_shape, sobolev_space, mapping
        ) = canonical_element_description(family, cell, degree, form_degree)

        # TODO: Move these to base? Might be better to instead
        # simplify base though.
        self._sobolev_space = sobolev_space
        self._mapping = mapping
        self._short_name = short_name or family
        self._variant = variant

        # Type check variant
        if variant is not None and not isinstance(variant, str):
            raise ValueError("Illegal variant: must be string or None")

        # Initialize element data
        FiniteElementBase.__init__(self, family, cell, degree, quad_scheme,
                                   value_shape, reference_value_shape)

        # Cache repr string
        qs = self.quadrature_scheme()
        if qs is None:
            quad_str = ""
        else:
            quad_str = ", quad_scheme=%s" % repr(qs)
        v = self.variant()
        if v is None:
            var_str = ""
        else:
            var_str = ", variant=%s" % repr(v)
        self._repr = "FiniteElement(%s, %s, %s%s%s)" % (
            repr(self.family()), repr(self.cell), repr(self.degree()), quad_str, var_str)
        assert '"' not in self._repr

    def __repr__(self):
        """Format as string for evaluation as Python object."""
        return self._repr

    def _is_globally_constant(self):
        """Doc."""
        return self.family() == "Real"

    def _is_linear(self):
        """Doc."""
        return self.family() == "Lagrange" and self.degree() == 1

    def mapping(self):
        """Return the mapping type for this element ."""
        return self._mapping

    @property
    def sobolev_space(self):
        """Return the underlying Sobolev space."""
        return self._sobolev_space

    def variant(self):
        """Return the variant used to initialise the element."""
        return self._variant

    def reconstruct(self, family=None, cell=None, degree=None, quad_scheme=None, variant=None):
        """Construct a new FiniteElement object with some properties replaced with new values."""
        if family is None:
            family = self.family()
        if cell is None:
            cell = self.cell
        if degree is None:
            degree = self.degree()
        if quad_scheme is None:
            quad_scheme = self.quadrature_scheme()
        if variant is None:
            variant = self.variant()
        return FiniteElement(family, cell, degree, quad_scheme=quad_scheme, variant=variant)

    def __str__(self):
        """Format as string for pretty printing."""
        qs = self.quadrature_scheme()
        qs = "" if qs is None else "(%s)" % qs
        v = self.variant()
        v = "" if v is None else "(%s)" % v
        return "<%s%s%s%s on a %s>" % (self._short_name, istr(self.degree()),
                                       qs, v, self.cell)

    def shortstr(self):
        """Format as string for pretty printing."""
        return f"{self._short_name}{istr(self.degree())}({self.quadrature_scheme()},{istr(self.variant())})"

    def __getnewargs__(self):
        """Return the arguments which pickle needs to recreate the object."""
        return (self.family(),
                self.cell,
                self.degree(),
                None,
                self.quadrature_scheme(),
                self.variant())

    @property
    def embedded_subdegree(self):
        """Return embedded subdegree."""
        if isinstance(self.degree(), int):
            return self.degree()
        else:
            return min(self.degree())

    @property
    def embedded_superdegree(self):
        """Return embedded superdegree."""
        if isinstance(self.degree(), int):
            return self.degree()
        else:
            return max(self.degree())
