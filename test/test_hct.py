import FIAT
import finat
import numpy as np
import pytest
from itertools import chain
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
    sd = ref_complex.get_spatial_dimension()
    top = ref_complex.get_topology()
    pts = []
    for cell in top[sd]:
        pts.extend(ref_complex.make_points(sd, cell, degree+sd+1))
    return pts


@pytest.mark.parametrize("reduced", (False, True), ids=("standard", "reduced"))
def test_hct(ref_cell, phys_cell, reduced):
    degree = 3
    if reduced:
        ref_element = finat.ReducedHsiehCloughTocher(ref_cell, degree)
    else:
        ref_element = finat.HsiehCloughTocher(ref_cell, degree, avg=True)
    ref_pts = finat.point_set.PointSet(make_unisolvent_points(ref_element._element))

    mapping = MyMapping(ref_cell, phys_cell)
    z = (0,) * ref_element.cell.get_spatial_dimension()
    finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr

    phys_element = FIAT.HsiehCloughTocher(phys_cell, degree, reduced=reduced)
    phys_points = make_unisolvent_points(phys_element)
    phys_vals = phys_element.tabulate(0, phys_points)[z]

    numbf = ref_element.space_dimension()
    assert np.allclose(finat_vals.T, phys_vals[:numbf])
