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


@pytest.fixture(params=["tet", "quad", "prism"], scope="module")
def cell(request):
    if request.param == "tet":
        cell = (FIAT.ufc_simplex(3),)
    elif request.param == "quad":
        interval = FIAT.ufc_simplex(1)
        cell = (interval, interval)
    elif request.param == "prism":
        triangle = FIAT.ufc_simplex(2)
        interval = FIAT.ufc_simplex(1)
        cell = (triangle, interval)
    return cell


@pytest.fixture
def ps(cell):
    dim = sum(e.get_spatial_dimension() for e in cell)
    return PointSet([[1/3, 1/4, 1/5][:dim]])


@pytest.fixture(scope="module")
def scalar_element(cell):
    if len(cell) == 1:
        return finat.Lagrange(cell[0], 4)
    else:
        e1, e2 = cell
        return finat.FlattenedDimensions(
            finat.TensorProductElement([
                finat.GaussLobattoLegendre(e1, 3),
                finat.GaussLobattoLegendre(e2, 3)]
            )
        )


@pytest.fixture(scope="module")
def hdiv_element(cell):
    if len(cell) == 1:
        return finat.RaviartThomas(cell[0], 3, variant="integral(3)")
    else:
        e1, e2 = cell
        element = finat.GaussLobattoLegendre if e1.get_spatial_dimension() == 1 else finat.RaviartThomas
        return finat.FlattenedDimensions(
            finat.EnrichedElement([
                finat.HDivElement(
                    finat.TensorProductElement([
                        element(e1, 3),
                        finat.GaussLegendre(e2, 3)])),
                finat.HDivElement(
                    finat.TensorProductElement([
                        finat.GaussLegendre(e1, 3),
                        finat.GaussLobattoLegendre(e2, 3)]))
            ]))


@pytest.fixture(scope="module")
def hcurl_element(cell):
    if len(cell) == 1:
        return finat.Nedelec(cell[0], 3, variant="integral(3)")
    else:
        e1, e2 = cell
        element = finat.GaussLegendre if e1.get_spatial_dimension() == 1 else finat.Nedelec
        return finat.FlattenedDimensions(
            finat.EnrichedElement([
                finat.HCurlElement(
                    finat.TensorProductElement([
                        finat.GaussLobattoLegendre(e1, 3),
                        finat.GaussLegendre(e2, 3)])),
                finat.HCurlElement(
                    finat.TensorProductElement([
                        element(e1, 3),
                        finat.GaussLobattoLegendre(e2, 3)]))
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
