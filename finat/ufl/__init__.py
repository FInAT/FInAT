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

import warnings as _warnings

from finat.ufl.brokenelement import BrokenElement
from finat.ufl.enrichedelement import EnrichedElement, NodalEnrichedElement
from finat.ufl.finiteelement import FiniteElement
from finat.ufl.finiteelementbase import FiniteElementBase
from finat.ufl.hdivcurl import HCurlElement, HDivElement, WithMapping
from finat.ufl.mixedelement import MixedElement, TensorElement, VectorElement
from finat.ufl.restrictedelement import RestrictedElement
from finat.ufl.tensorproductelement import TensorProductElement
