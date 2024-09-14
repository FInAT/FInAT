import FIAT
import finat
import numpy as np
import pytest
from gem.interpreter import evaluate

from fiat_mapping import MyMapping


@pytest.fixture
def ref_cell(request):
    K = FIAT.ufc_simplex(2)
    return K


@pytest.fixture
def phys_cell(request):
    K = FIAT.ufc_simplex(2)
    K.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    return K


def make_unisolvent_points(element):
    degree = element.degree()
    ref_complex = element.get_reference_complex()
    top = ref_complex.get_topology()
    pts = []
    for dim in top:
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, degree, variant="gll"))
    return pts


def check_zany_mapping(finat_element, phys_element):
    ref_element = finat_element.fiat_equivalent
    shape = ref_element.value_shape()

    ref_cell = ref_element.get_reference_element()
    phys_cell = phys_element.get_reference_element()
    mapping = MyMapping(ref_cell, phys_cell)

    ref_points = make_unisolvent_points(ref_element)
    ps = finat.point_set.PointSet(ref_points)

    z = (0,) * finat_element.cell.get_spatial_dimension()
    finat_vals_gem = finat_element.basis_evaluation(0, ps, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr.transpose(*range(1, len(shape)+2), 0)

    phys_points = make_unisolvent_points(phys_element)
    phys_vals = phys_element.tabulate(0, phys_points)[z]

    numdofs = finat_element.space_dimension()
    numbfs = phys_element.space_dimension()
    if numbfs == numdofs:
        # Solve for the basis transformation and compare results
        ref_vals = ref_element.tabulate(0, ref_points)[z]
        Phi = ref_vals.reshape(numbfs, -1)
        phi = phys_vals.reshape(numbfs, -1)
        Mh = np.linalg.solve(Phi @ Phi.T, Phi @ phi.T).T
        Mh[abs(Mh) < 1E-10] = 0

        Mgem = finat_element.basis_transformation(mapping)
        M = evaluate([Mgem])[0].arr
        assert np.allclose(M, Mh, atol=1E-9), str(Mh-M)

    assert np.allclose(finat_vals, phys_vals[:numdofs])


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         finat.QuadraticPowellSabin6,
                         finat.QuadraticPowellSabin12,
                         finat.Hermite,
                         finat.ReducedHsiehCloughTocher,
                         finat.Bell,
                         ])
def test_C1_elements(ref_cell, phys_cell, element):
    kwargs = {}
    finat_kwargs = {}
    if element == finat.ReducedHsiehCloughTocher:
        kwargs = dict(reduced=True)
    if element == finat.QuadraticPowellSabin12:
        finat_kwargs = dict(avg=True)
    finat_element = element(ref_cell, **finat_kwargs)
    phys_element = type(finat_element.fiat_equivalent)(phys_cell, **kwargs)
    check_zany_mapping(finat_element, phys_element)


@pytest.mark.parametrize("element, degree", [
    *((finat.Argyris, k) for k in range(5, 8)),
    *((finat.HsiehCloughTocher, k) for k in range(3, 6))
])
def test_high_order_C1_elements(ref_cell, phys_cell, element, degree):
    finat_element = element(ref_cell, degree, avg=True)
    phys_element = type(finat_element.fiat_equivalent)(phys_cell, degree)
    check_zany_mapping(finat_element, phys_element)


def test_argyris_point(ref_cell, phys_cell):
    finat_element = finat.Argyris(ref_cell, variant="point")
    phys_element = type(finat_element.fiat_equivalent)(phys_cell, variant="point")
    check_zany_mapping(finat_element, phys_element)


@pytest.mark.parametrize("element", [
                         finat.MardalTaiWinther,
                         finat.AlfeldSorokina,
                         ])
def test_piola_C0_elements(ref_cell, phys_cell, element):
    sd = ref_cell.get_spatial_dimension()
    z = (0,)*sd
    ref_el_finat = element(ref_cell)

    ref_element = ref_el_finat._element
    shape = ref_element.value_shape()
    ref_pts = make_unisolvent_points(ref_element)
    ref_vals = ref_element.tabulate(0, ref_pts)[z]

    phys_element = type(ref_element)(phys_cell)

    phys_pts = make_unisolvent_points(phys_element)
    phys_vals = phys_element.tabulate(0, phys_pts)[z]

    # Piola map the reference elements
    J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                      phys_cell.vertices)
    detJ = np.linalg.det(J)
    K = J / detJ

    ref_vals_piola = np.zeros(ref_vals.shape)
    for i in range(ref_vals.shape[0]):
        for k in range(ref_vals.shape[-1]):
            ref_vals_piola[i, ..., k] = K @ ref_vals[i, ..., k]

    num_dofs = ref_el_finat.space_dimension()
    num_bfs = phys_element.space_dimension()
    ids = ref_el_finat.entity_dofs()
    num_facet_bfs = sum(len(ids[dim][entity])
                        for dim in ids if dim < sd
                        for entity in ids[dim])

    # Zany map the results
    mappng = MyMapping(ref_cell, phys_cell)
    Mgem = ref_el_finat.basis_transformation(mappng)
    M = evaluate([Mgem])[0].arr
    shp = (num_dofs, *shape, -1)
    ref_vals_zany = (M @ ref_vals_piola.reshape(num_bfs, -1)).reshape(shp)

    # Solve for the basis transformation and compare results
    Phi = ref_vals_piola.reshape(num_bfs, -1)
    phi = phys_vals.reshape(num_bfs, -1)
    Mh = np.linalg.solve(Phi @ Phi.T, Phi @ phi.T).T
    Mh[abs(Mh) < 1E-10] = 0.0
    assert np.allclose(M[:num_facet_bfs], Mh[:num_facet_bfs]), str(M-Mh)

    assert np.allclose(ref_vals_zany[:num_facet_bfs], phys_vals[:num_facet_bfs])
