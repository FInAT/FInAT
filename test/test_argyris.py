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


def run_argyris(ref_cell, phys_cell, degree, variant):
    ref_element = finat.Argyris(ref_cell, degree, variant=variant, avg=True)
    ref_pts = finat.point_set.PointSet(make_unisolvent_points(ref_element._element))

    mapping = MyMapping(ref_cell, phys_cell)
    z = (0,) * ref_element.cell.get_spatial_dimension()
    finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr.T

    phys_element = FIAT.Argyris(phys_cell, degree, variant=variant)
    phys_points = make_unisolvent_points(phys_element)
    phys_vals = phys_element.tabulate(0, phys_points)[z]
    numbfs = ref_element.space_dimension()

    # Solve for the basis transformation and compare results
    Mgem = ref_element.basis_transformation(mapping)
    M = evaluate([Mgem])[0].arr
    ref_vals = ref_element._element.tabulate(0, ref_pts.points)[z]
    Phi = ref_vals.reshape(numbfs, -1)
    phi = phys_vals.reshape(numbfs, -1)
    Mh = np.linalg.solve(Phi @ Phi.T, Phi @ phi.T).T
    Mh[abs(Mh) < 1E-10] = 0
    # edofs = ref_element.entity_dofs()
    # i = len(edofs[2][0])
    # offset = len(edofs[1][0]) + i
    # print()
    # print(Mh.T[-offset:-i])
    # print(M.T[-offset:-i])
    assert np.allclose(M, Mh, atol=1E-9)

    assert np.allclose(finat_vals, phys_vals[:numbfs])


def test_point_argyris(ref_cell, phys_cell):
    run_argyris(ref_cell, phys_cell, 5, "point")


@pytest.mark.parametrize("degree", range(5, 8))
def test_integral_argyris(ref_cell, phys_cell, degree):
    run_argyris(ref_cell, phys_cell, degree, "integral")
