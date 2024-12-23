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

            family, short_name, degree, reference_value_shape, sobolev_space, mapping, embedded_degree = \
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

            elif family == "NCF":
                cell_h, cell_v = cell.sub_cells()
                if cell_h.cellname() != "quadrilateral":
                    raise ValueError(f"{family} is available on TensorProductCell(quadrilateral, interval) only.")
                if cell_v.cellname() != "interval":
                    raise ValueError(f"{family} is available on TensorProductCell(quadrilateral, interval) only.")

                Qc_elt = FiniteElement("RTCF", "quadrilateral", degree, variant=variant)
                Qd_elt = FiniteElement("DQ", "quadrilateral", degree - 1, variant=variant)

                Id_elt = FiniteElement("DG", "interval", degree - 1, variant=variant)
                Ic_elt = FiniteElement("CG", "interval", degree, variant=variant)

                return EnrichedElement(HDiv(TensorProductElement(Qc_elt, Id_elt, cell=cell)),
                                       HDiv(TensorProductElement(Qd_elt, Ic_elt, cell=cell)))

            elif family == "NCE":
                cell_h, cell_v = cell.sub_cells()
                if cell_h.cellname() != "quadrilateral":
                    raise ValueError(f"{family} is available on TensorProductCell(quadrilateral, interval) only.")
                if cell_v.cellname() != "interval":
                    raise ValueError(f"{family} is available on TensorProductCell(quadrilateral, interval) only.")

                Qc_elt = FiniteElement("Q", "quadrilateral", degree, variant=variant)
                Qd_elt = FiniteElement("RTCE", "quadrilateral", degree, variant=variant)

                Id_elt = FiniteElement("DG", "interval", degree - 1, variant=variant)
                Ic_elt = FiniteElement("CG", "interval", degree, variant=variant)

                return EnrichedElement(HCurl(TensorProductElement(Qc_elt, Id_elt, cell=cell)),
                                       HCurl(TensorProductElement(Qd_elt, Ic_elt, cell=cell)))

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

        return super().__new__(cls)

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
            family, short_name, degree, reference_value_shape, sobolev_space, mapping, embedded_degree
        ) = canonical_element_description(family, cell, degree, form_degree)

        # TODO: Move these to base? Might be better to instead
        # simplify base though.
        self._sobolev_space = sobolev_space
        self._mapping = mapping
        self._short_name = short_name or family
        self._variant = variant
        self._embedded_degree = embedded_degree

        # Type check variant
        if variant is not None and not isinstance(variant, str):
            raise ValueError("Illegal variant: must be string or None")

        # Initialize element data
        FiniteElementBase.__init__(self, family, cell, degree, quad_scheme,
                                   reference_value_shape)

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
        subdegree = self.degree()
        if not isinstance(subdegree, int):
            subdegree = min(subdegree)
        if isinstance(self._embedded_degree, int):
            subdegree = min(subdegree, self._embedded_degree)
        return subdegree

    @property
    def embedded_superdegree(self):
        """Return embedded superdegree."""
        superdegree = self.degree()
        if not isinstance(superdegree, int):
            superdegree = max(superdegree)
        if isinstance(self._embedded_degree, int):
            superdegree = max(superdegree, self._embedded_degree)
        return superdegree
