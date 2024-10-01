import FIAT
import finat
import numpy as np
import pytest
from gem.interpreter import evaluate

from fiat_mapping import MyMapping
from finat import *


@pytest.fixture
def ref_cell(request):
    K = FIAT.ufc_simplex(2)
    return K


@pytest.fixture
def phys_cell(request):
    K = FIAT.ufc_simplex(2)
    K.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    return K


def make_unisolvent_points(element, interior=False):
    degree = element.degree()
    ref_complex = element.get_reference_complex()
    top = ref_complex.get_topology()
    pts = []
    if interior:
        dim = ref_complex.get_spatial_dimension()
        for entity in top[dim]:
            pts.extend(ref_complex.make_points(dim, entity, degree+dim+1, variant="gll"))
    else:
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
        assert np.allclose(M, Mh, atol=1E-9), str(Mh.T-M.T)

    assert np.allclose(finat_vals, phys_vals[:numdofs])


@pytest.mark.parametrize("element", [
                         Morley,
                         QuadraticPowellSabin6,
                         QuadraticPowellSabin12,
                         Hermite,
                         ReducedHsiehCloughTocher,
                         Bell,
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


def check_zany_piola_mapping(finat_element, phys_element):
    ref_element = finat_element._element
    ref_cell = ref_element.get_reference_element()
    phys_cell = phys_element.get_reference_element()
    sd = ref_cell.get_spatial_dimension()
    try:
        indices = finat_element._indices
    except AttributeError:
        indices = slice(None)

    shape = ref_element.value_shape()
    ref_pts = make_unisolvent_points(ref_element, interior=True)
    ref_vals = ref_element.tabulate(0, ref_pts)[(0,)*sd]

    phys_pts = make_unisolvent_points(phys_element, interior=True)
    phys_vals = phys_element.tabulate(0, phys_pts)[(0,)*sd]

    # Piola map the reference elements
    J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                      phys_cell.vertices)
    detJ = np.linalg.det(J)
    K = J / detJ
    if len(shape) == 2:
        piola_map = lambda x: K @ x @ K.T
    else:
        piola_map = lambda x: K @ x

    ref_vals_piola = np.zeros(ref_vals.shape)
    for i in range(ref_vals.shape[0]):
        for k in range(ref_vals.shape[-1]):
            ref_vals_piola[i, ..., k] = piola_map(ref_vals[i, ..., k])

    dofs = finat_element.entity_dofs()
    num_bfs = phys_element.space_dimension()
    num_dofs = finat_element.space_dimension()
    num_facet_dofs = num_dofs - sum(len(dofs[sd][entity]) for entity in dofs[sd])

    # Zany map the results
    mappng = MyMapping(ref_cell, phys_cell)
    Mgem = finat_element.basis_transformation(mappng)
    M = evaluate([Mgem])[0].arr
    shp = (num_dofs, *shape, -1)
    ref_vals_zany = (M @ ref_vals_piola.reshape(num_bfs, -1)).reshape(shp)

    # Solve for the basis transformation and compare results
    Phi = ref_vals_piola.reshape(num_bfs, -1)
    phi = phys_vals.reshape(num_bfs, -1)
    Mh = np.linalg.solve(Phi @ Phi.T, Phi @ phi.T).T
    M = M[:num_facet_dofs]
    Mh = Mh[indices][:num_facet_dofs]
    Mh[abs(Mh) < 1E-10] = 0.0
    M[abs(M) < 1E-10] = 0.0
    assert np.allclose(M, Mh), str(M.T - Mh.T)
    assert np.allclose(ref_vals_zany[:num_facet_dofs], phys_vals[indices][:num_facet_dofs])


@pytest.mark.parametrize("element", [
                         MardalTaiWinther,
                         BernardiRaugel,
                         ArnoldQin,
                         ReducedArnoldQin,
                         AlfeldSorokina,
                         ArnoldWinther,
                         ArnoldWintherNC,
                         JohnsonMercier,
                         ])
def test_piola_triangle(ref_cell, phys_cell, element):
    finat_element = element(ref_cell)
    phys_element = type(finat_element.fiat_equivalent)(phys_cell)
    check_zany_piola_mapping(finat_element, phys_element)


@pytest.fixture
def ref_tet(request):
    K = FIAT.ufc_simplex(3)
    return K


@pytest.fixture
def phys_tet(request):
    K = FIAT.ufc_simplex(3)
    K.vertices = ((0, 0, 0),
                  (1., 0.1, -0.37),
                  (0.01, 0.987, -.23),
                  (-0.1, -0.2, 1.38))
    return K


@pytest.mark.parametrize("element", [
                         BernardiRaugel,
                         ChristiansenHu,
                         AlfeldSorokina,
                         JohnsonMercier,
                         ])
def test_piola_tetrahedron(ref_tet, phys_tet, element):
    finat_element = element(ref_tet)
    phys_element = type(finat_element.fiat_equivalent)(phys_tet)
    check_zany_piola_mapping(finat_element, phys_element)
