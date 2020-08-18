import FIAT
import finat
import numpy
import pytest
from finat.point_set import PointSet
from finat.restricted import r_to_codim
from gem.interpreter import evaluate


def tabulate(element, ps):
    tabulation, = element.basis_evaluation(0, ps).values()
    result, = evaluate([tabulation])
    # Singleton point
    shape = (int(numpy.prod(element.index_shape)), ) + element.value_shape
    return result.arr.reshape(*shape)


def which_dofs(element, restricted):
    edofs = element.entity_dofs()
    rdofs = restricted.entity_dofs()
    keep_e = []
    keep_r = []
    for k in edofs.keys():
        for e, indices in edofs[k].items():
            if rdofs[k][e]:
                assert len(rdofs[k][e]) == len(indices)
                keep_e.extend(indices)
                keep_r.extend(rdofs[k][e])
    return keep_e, keep_r


@pytest.fixture(params=["vertex", "edge", "facet", "interior"], scope="module")
def restriction(request):
    return request.param


@pytest.fixture(params=["tet", "quad"], scope="module")
def cell(request):
    return request.param


@pytest.fixture
def ps(cell):
    if cell == "tet":
        return PointSet([[1/3, 1/4, 1/5]])
    elif cell == "quad":
        return PointSet([[1/3, 1/4]])


@pytest.fixture(scope="module")
def scalar_element(cell):
    if cell == "tet":
        return finat.Lagrange(FIAT.reference_element.UFCTetrahedron(), 4)
    elif cell == "quad":
        interval = FIAT.reference_element.UFCInterval()
        return finat.FlattenedDimensions(
            finat.TensorProductElement([
                finat.GaussLobattoLegendre(interval, 3),
                finat.GaussLobattoLegendre(interval, 3)]
            )
        )


@pytest.fixture(scope="module")
def hdiv_element(cell):
    if cell == "tet":
        return finat.RaviartThomas(FIAT.reference_element.UFCTetrahedron(), 3, variant="integral(3)")
    elif cell == "quad":
        interval = FIAT.reference_element.UFCInterval()
        return finat.FlattenedDimensions(
            finat.EnrichedElement([
                finat.HDivElement(
                    finat.TensorProductElement([
                        finat.GaussLobattoLegendre(interval, 3),
                        finat.GaussLegendre(interval, 3)])),
                finat.HDivElement(
                    finat.TensorProductElement([
                        finat.GaussLegendre(interval, 3),
                        finat.GaussLobattoLegendre(interval, 3)]))
            ]))


@pytest.fixture(scope="module")
def hcurl_element(cell):
    if cell == "tet":
        return finat.Nedelec(FIAT.reference_element.UFCTetrahedron(), 3, variant="integral(3)")
    elif cell == "quad":
        interval = FIAT.reference_element.UFCInterval()
        return finat.FlattenedDimensions(
            finat.EnrichedElement([
                finat.HCurlElement(
                    finat.TensorProductElement([
                        finat.GaussLobattoLegendre(interval, 3),
                        finat.GaussLegendre(interval, 3)])),
                finat.HCurlElement(
                    finat.TensorProductElement([
                        finat.GaussLegendre(interval, 3),
                        finat.GaussLobattoLegendre(interval, 3)]))
            ]))


def run_restriction(element, restriction, ps):
    try:
        restricted = finat.RestrictedElement(element, restriction)
    except ValueError:
        # No dofs.
        # Check that the original element had no dofs in all the relevant slots.
        dim = element.cell.get_spatial_dimension()
        lo_codim = r_to_codim(restriction, dim)
        hi_codim = (lo_codim if restriction == "interior" else dim)
        edofs = element.entity_dofs()
        for entity_dim, dof_numbering in edofs.items():
            try:
                entity_codim = dim - sum(entity_dim)
            except TypeError:
                entity_codim = dim - entity_dim
            if lo_codim <= entity_codim <= hi_codim:
                assert all(len(i) == 0 for i in dof_numbering.values())
    else:
        e = tabulate(element, ps)
        r = tabulate(restricted, ps)
        keep_e, keep_r = which_dofs(element, restricted)
        assert numpy.allclose(e[keep_e, ...], r[keep_r, ...])


def test_scalar_restriction(scalar_element, restriction, ps):
    run_restriction(scalar_element, restriction, ps)


def test_hdiv_restriction(hdiv_element, restriction, ps):
    run_restriction(hdiv_element, restriction, ps)


def test_hcurl_restriction(hcurl_element, restriction, ps):
    run_restriction(hcurl_element, restriction, ps)
