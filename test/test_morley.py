import FIAT
import finat
import numpy as np
from gem.interpreter import evaluate
from fiat_mapping import MyMapping


def test_morley():
    ref_cell = FIAT.ufc_simplex(2)
    ref_element = finat.Morley(ref_cell, 2)
    ref_pts = finat.point_set.PointSet(ref_cell.make_points(2, 0, 4))

    phys_cell = FIAT.ufc_simplex(2)
    phys_cell.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))

    mppng = MyMapping(ref_cell, phys_cell)
    z = (0, 0)
    finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mppng)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr

    phys_cell_FIAT = FIAT.Morley(phys_cell)
    phys_points = phys_cell.make_points(2, 0, 4)
    phys_vals = phys_cell_FIAT.tabulate(0, phys_points)[z]

    assert np.allclose(finat_vals, phys_vals.T)
