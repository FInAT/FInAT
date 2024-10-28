import FIAT
import finat
import numpy as np
import pytest
from gem.interpreter import evaluate

from fiat_mapping import MyMapping


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


def check_zany_mapping(element, ref_cell, phys_cell, *args, **kwargs):
    phys_element = element(phys_cell, *args, **kwargs).fiat_equivalent
    finat_element = element(ref_cell, *args, **kwargs)

    ref_element = finat_element._element
    ref_cell = ref_element.get_reference_element()
    phys_cell = phys_element.get_reference_element()
    sd = ref_cell.get_spatial_dimension()

    shape = ref_element.value_shape()
    ref_pts = make_unisolvent_points(ref_element, interior=True)
    ref_vals = ref_element.tabulate(0, ref_pts)[(0,)*sd]

    phys_pts = make_unisolvent_points(phys_element, interior=True)
    phys_vals = phys_element.tabulate(0, phys_pts)[(0,)*sd]

    mapping = ref_element.mapping()[0]
    if mapping == "affine":
        ref_vals_piola = ref_vals
    else:
        # Piola map the reference elements
        J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                          phys_cell.vertices)
        K = []
        if "covariant" in mapping:
            K.append(np.linalg.inv(J).T)
        if "contravariant" in mapping:
            K.append(J / np.linalg.det(J))

        if len(shape) == 2:
            piola_map = lambda x: K[0] @ x @ K[-1].T
        else:
            piola_map = lambda x: K[0] @ x

        ref_vals_piola = np.zeros(ref_vals.shape)
        for i in range(ref_vals.shape[0]):
            for k in range(ref_vals.shape[-1]):
                ref_vals_piola[i, ..., k] = piola_map(ref_vals[i, ..., k])

    # Zany map the results
    num_bfs = phys_element.space_dimension()
    num_dofs = finat_element.space_dimension()
    mappng = MyMapping(ref_cell, phys_cell)
    try:
        Mgem = finat_element.basis_transformation(mappng)
        M = evaluate([Mgem])[0].arr
        ref_vals_zany = np.tensordot(M, ref_vals_piola, (-1, 0))
    except AttributeError:
        M = np.eye(num_dofs, num_bfs)
        ref_vals_zany = ref_vals_piola

    # Solve for the basis transformation and compare results
    Phi = ref_vals_piola.reshape(num_bfs, -1)
    phi = phys_vals.reshape(num_bfs, -1)
    Mh = np.linalg.solve(Phi @ Phi.T, Phi @ phi.T).T
    Mh = Mh[:num_dofs]
    Mh[abs(Mh) < 1E-10] = 0.0
    M[abs(M) < 1E-10] = 0.0
    assert np.allclose(M, Mh), str(M.T - Mh.T)
    assert np.allclose(ref_vals_zany, phys_vals[:num_dofs])


@pytest.fixture
def ref_tri(request):
    K = FIAT.ufc_simplex(2)
    return K


@pytest.fixture
def phys_tri(request):
    K = FIAT.ufc_simplex(2)
    K.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    return K


@pytest.mark.parametrize("element", [
                         finat.Morley,
                         finat.QuadraticPowellSabin6,
                         finat.QuadraticPowellSabin12,
                         finat.Hermite,
                         finat.ReducedHsiehCloughTocher,
                         finat.Bell,
                         ])
def test_C1_elements(ref_tri, phys_tri, element):
    kwargs = {}
    if element == finat.QuadraticPowellSabin12:
        kwargs = dict(avg=True)
    check_zany_mapping(element, ref_tri, phys_tri, **kwargs)


@pytest.mark.parametrize("element, degree", [
    *((finat.Argyris, k) for k in range(5, 8)),
    *((finat.HsiehCloughTocher, k) for k in range(3, 6))
])
def test_high_order_C1_elements(ref_tri, phys_tri, element, degree):
    check_zany_mapping(element, ref_tri, phys_tri, degree, avg=True)


def test_argyris_point(ref_tri, phys_tri):
    check_zany_mapping(finat.Argyris, ref_tri, phys_tri, variant="point")


@pytest.mark.parametrize("element", [
                         finat.MardalTaiWinther,
                         finat.BernardiRaugel,
                         finat.BernardiRaugelBubble,
                         finat.ReducedArnoldQin,
                         finat.AlfeldSorokina,
                         finat.ChristiansenHu,
                         finat.ArnoldWinther,
                         finat.ArnoldWintherNC,
                         finat.JohnsonMercier,
                         finat.GuzmanNeilanFirstKindH1,
                         finat.GuzmanNeilanSecondKindH1,
                         finat.GuzmanNeilanBubble,
                         ])
def test_piola_triangle(ref_tri, phys_tri, element):
    check_zany_mapping(element, ref_tri, phys_tri)


@pytest.mark.parametrize("element, degree, variant", [
    *((finat.HuZhang, k, v) for v in ("integral", "point") for k in range(3, 6)),
])
def test_piola_triangle_high_order(ref_tri, phys_tri, element, degree, variant):
    check_zany_mapping(element, ref_tri, phys_tri, degree, variant)


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
                         finat.BernardiRaugel,
                         finat.BernardiRaugelBubble,
                         finat.ChristiansenHu,
                         finat.AlfeldSorokina,
                         finat.JohnsonMercier,
                         finat.GuzmanNeilanFirstKindH1,
                         finat.GuzmanNeilanSecondKindH1,
                         finat.GuzmanNeilanBubble,
                         finat.GuzmanNeilanH1div,
                         ])
def test_piola_tetrahedron(ref_tet, phys_tet, element):
    check_zany_mapping(element, ref_tet, phys_tet)


@pytest.mark.parametrize("element, degree", [
                         *((finat.Regge, k) for k in range(3)),
                         *((finat.HellanHerrmannJohnson, k) for k in range(3)),
                         ])
@pytest.mark.parametrize("dimension", [2, 3])
def test_affine(ref_tri, phys_tri, ref_tet, phys_tet, element, degree, dimension):
    if dimension == 2:
        ref_el, phys_el = ref_tri, phys_tri
    elif dimension == 3:
        ref_el, phys_el = ref_tet, phys_tet
    check_zany_mapping(element, ref_el, phys_el, degree)
