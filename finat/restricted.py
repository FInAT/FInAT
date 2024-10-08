from functools import singledispatch
from itertools import chain

import FIAT
from FIAT.polynomial_set import mis

import finat
from finat.fiat_elements import FiatElement
from finat.physically_mapped import PhysicallyMappedElement


# Sentinel for when restricted element is empty
null_element = object()


@singledispatch
def restrict(element, domain, take_closure):
    """Restrict an element to a given subentity.

    :arg element: The element to restrict.
    :arg domain: The subentity to restrict to.
    :arg take_closure: Gather dofs in closure of the subentities?
        Ignored for "interior" domain.

    :raises NotImplementedError: If we don't know how to restrict this
        element.
    :raises ValueError: If the restricted element is empty.
    :returns: A new finat element."""
    return NotImplementedError(f"Don't know how to restrict element of type {type(element)}")


@restrict.register(FiatElement)
def restrict_fiat(element, domain, take_closure):
    try:
        return FiatElement(FIAT.RestrictedElement(element._element,
                           restriction_domain=domain, take_closure=take_closure))
    except ValueError:
        return null_element


@restrict.register(PhysicallyMappedElement)
def restrict_physically_mapped(element, domain, take_closure):
    raise NotImplementedError("Can't restrict Physically Mapped things")


@restrict.register(finat.FlattenedDimensions)
def restrict_flattened_dimensions(element, domain, take_closure):
    restricted = restrict(element.product, domain, take_closure)
    if restricted is null_element:
        return null_element
    else:
        return finat.FlattenedDimensions(restricted)


@restrict.register(finat.DiscontinuousElement)
@restrict.register(finat.DiscontinuousLagrange)
@restrict.register(finat.Legendre)
def restrict_discontinuous(element, domain, take_closure):
    if domain == "interior":
        return element
    else:
        return null_element


@restrict.register(finat.EnrichedElement)
def restrict_enriched(element, domain, take_closure):
    if all(isinstance(e, finat.mixed.MixedSubElement) for e in element.elements):
        # Mixed is handled by Enriched + MixedSubElement, we must
        # restrict the subelements here because the transformation is
        # nonlocal.
        elements = tuple(restrict(e.element, domain, take_closure) for
                         e in element.elements)
        reconstruct = finat.mixed.MixedElement
    elif not any(isinstance(e, finat.mixed.MixedSubElement) for e in element.elements):
        elements = tuple(restrict(e, domain, take_closure)
                         for e in element.elements)
        reconstruct = finat.EnrichedElement
    else:
        raise NotImplementedError("Not expecting enriched with mixture of MixedSubElement and others")

    elements = tuple(e for e in elements if e is not null_element)
    if elements:
        return reconstruct(elements)
    else:
        return null_element


@restrict.register(finat.HCurlElement)
def restrict_hcurl(element, domain, take_closure):
    restricted = restrict(element.wrappee, domain, take_closure)
    if restricted is null_element:
        return null_element
    else:
        if isinstance(restricted, finat.EnrichedElement):
            return finat.EnrichedElement(finat.HCurlElement(e)
                                         for e in restricted.elements)
        else:
            return finat.HCurlElement(restricted)


@restrict.register(finat.HDivElement)
def restrict_hdiv(element, domain, take_closure):
    restricted = restrict(element.wrappee, domain, take_closure)
    if restricted is null_element:
        return null_element
    else:
        if isinstance(restricted, finat.EnrichedElement):
            return finat.EnrichedElement(finat.HDivElement(e)
                                         for e in restricted.elements)
        else:
            return finat.HDivElement(restricted)


@restrict.register(finat.mixed.MixedSubElement)
def restrict_mixed(element, domain, take_closure):
    raise AssertionError("Was expecting this to be handled inside EnrichedElement restriction")


def r_to_codim(restriction, dim):
    if restriction == "interior":
        return 0
    elif restriction == "facet":
        return 1
    elif restriction == "face":
        return dim - 2
    elif restriction == "edge":
        return dim - 1
    elif restriction == "vertex":
        return dim
    else:
        raise ValueError


def codim_to_r(codim, dim):
    d = dim - codim
    if codim == 0:
        return "interior"
    elif codim == 1:
        return "facet"
    elif d == 0:
        return "vertex"
    elif d == 1:
        return "edge"
    elif d == 2:
        return "face"
    else:
        raise ValueError


@restrict.register(finat.TensorProductElement)
def restrict_tpe(element, domain, take_closure):
    # The restriction of a TPE to a codim subentity is the direct sum
    # of TPEs where the factors have been restricted in such a way
    # that the sum of those restrictions is codim.
    #
    # For example, to restrict an interval x interval to edges (codim 1)
    # we construct
    #
    # R(I, 0)⊗R(I, 1) ⊕ R(I, 1)⊗R(I, 0)
    #
    # If take_closure is true, the restriction wants to select dofs on
    # entities with dim >= codim >= 1 (for the edge example)
    # so we get
    #
    # R(I, 0)⊗R(I, 1) ⊕ R(I, 1)⊗R(I, 0) ⊕ R(I, 0)⊗R(I, 0)
    factors = element.factors
    dimension = element.cell.get_spatial_dimension()
    # Figure out which codim entity we're selecting
    codim = r_to_codim(domain, dimension)
    # And the range of codims.
    upper = 1 + (dimension
                 if (take_closure and domain != "interior")
                 else codim)
    # restrictions on each factor taken from n-tuple that sums to the
    # target codim (as long as the codim <= dim_factor)
    restrictions = tuple(candidate
                         for candidate in chain(*(mis(len(factors), c)
                                                  for c in range(codim, upper)))
                         if all(d <= factor.cell.get_dimension()
                                for d, factor in zip(candidate, factors)))
    take_closure = False
    elements = []
    for decomposition in restrictions:
        # Recurse, but don't take closure in recursion (since we
        # handled it already).
        new_factors = tuple(
            restrict(factor, codim_to_r(codim, factor.cell.get_dimension()),
                     take_closure)
            for factor, codim in zip(factors, decomposition))
        # If one of the factors was empty then the whole TPE is empty,
        # so skip.
        if all(f is not null_element for f in new_factors):
            elements.append(finat.TensorProductElement(new_factors))
    if elements:
        return finat.EnrichedElement(elements)
    else:
        return null_element


@restrict.register(finat.TensorFiniteElement)
def restrict_tfe(element, domain, take_closure):
    restricted = restrict(element._base_element, domain, take_closure)
    if restricted is null_element:
        return null_element
    else:
        return finat.TensorFiniteElement(restricted, element._shape, element._transpose)


@restrict.register(finat.HDivTrace)
def restrict_hdivtrace(element, domain, take_closure):
    try:
        return FiatElement(FIAT.RestrictedElement(element._element, restriction_domain=domain))
    except ValueError:
        return null_element


def RestrictedElement(element, restriction_domain, *, indices=None):
    """Construct a restricted element.

    :arg element: The element to restrict.
    :arg restriction_domain: Which entities to restrict to.
    :arg indices: Indices of basis functions to select (not supported)
    :returns: A new element.

    .. note::

       A restriction domain of "interior" means to select the dofs on
       the cell, all other domains (e.g. "face", "edge") select dofs
       in the closure of the entity.

    .. warning::

       The returned element *may not* be interpolatory. That is, the
       dual basis (if implemented) might not be nodal to the primal
       basis. Assembly still works (``basis_evaluation`` is fine), but
       interpolation may produce bad results.

       Restrictions of FIAT-implemented CiarletElements are always
       nodal.
    """
    if indices is not None:
        raise NotImplementedError("Only done for topological restrictions")
    assert restriction_domain is not None
    restricted = restrict(element, restriction_domain, take_closure=True)
    if restricted is null_element:
        raise ValueError("Restricted element is empty")
    return restricted
