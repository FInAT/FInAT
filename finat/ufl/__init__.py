"""Legacy UFL features."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file was originally part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Andrew T. T. McRae 2014
# Modified by Lawrence Mitchell 2014
# Modified by Matthew Scroggs, 2023

from finat.ufl.brokenelement import BrokenElement  # noqa: F401
from finat.ufl.enrichedelement import EnrichedElement, NodalEnrichedElement  # noqa: F401
from finat.ufl.finiteelement import FiniteElement  # noqa: F401
from finat.ufl.finiteelementbase import FiniteElementBase  # noqa: F401
from finat.ufl.hdivcurl import HCurlElement, HDivElement, WithMapping  # noqa: F401
from finat.ufl.mixedelement import MixedElement, TensorElement, VectorElement  # noqa: F401
from finat.ufl.restrictedelement import RestrictedElement  # noqa: F401
from finat.ufl.tensorproductelement import TensorProductElement  # noqa: F401
