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


def run_test(finat_element, phys_element):
    ref_element = finat_element.fiat_equivalent

    ref_cell = ref_element.get_reference_element()
    phys_cell = phys_element.get_reference_element()
    mapping = MyMapping(ref_cell, phys_cell)

    ref_points = make_unisolvent_points(ref_element)
    ps = finat.point_set.PointSet(ref_points)

    z = (0,) * finat_element.cell.get_spatial_dimension()
    finat_vals_gem = finat_element.basis_evaluation(0, ps, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr.T

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
        # edofs = finat_element.entity_dofs()
        # i = len(edofs[2][0])
        # offset = len(edofs[1][0]) + i
        # print()
        # print(Mh.T[-offset:-i])
        # print(M.T[-offset:-i])

        Mgem = finat_element.basis_transformation(mapping)
        M = evaluate([Mgem])[0].arr
        assert np.allclose(M, Mh, atol=1E-9)

    assert np.allclose(finat_vals, phys_vals[:numdofs])


@pytest.mark.parametrize("element, degree", [
                         (finat.Morley, 2),
                         (finat.Hermite, 3),
                         (finat.ReducedHsiehCloughTocher, 3),
                         (finat.Bell, 5)])
def test_C1_elements(ref_cell, phys_cell, element, degree):
    kwargs = {}
    if element == finat.ReducedHsiehCloughTocher:
        kwargs = dict(reduced=True)
    finat_element = element(ref_cell, degree)
    phys_element = type(finat_element.fiat_equivalent)(phys_cell, **kwargs)
    run_test(finat_element, phys_element)


def test_argyris_point(ref_cell, phys_cell):
    degree = 5
    finat_element = finat.Argyris(ref_cell, degree, variant="point")
    phys_element = type(finat_element.fiat_equivalent)(phys_cell, degree, variant="point")
    run_test(finat_element, phys_element)


@pytest.mark.parametrize("degree", range(5, 8))
def test_argyris_integral(ref_cell, phys_cell, degree):
    finat_element = finat.Argyris(ref_cell, degree, avg=True)
    phys_element = type(finat_element.fiat_equivalent)(phys_cell, degree)
    run_test(finat_element, phys_element)


@pytest.mark.parametrize("degree", range(3, 6))
def test_hct(ref_cell, phys_cell, degree):
    finat_element = finat.HsiehCloughTocher(ref_cell, degree, avg=True)
    phys_element = type(finat_element.fiat_equivalent)(phys_cell, degree)
    run_test(finat_element, phys_element)
