import FIAT
import finat
import numpy as np
from gem.interpreter import evaluate

from fiat_mapping import MyMapping

# def test_hct():
#     ref_cell = FIAT.ufc_simplex(2)
#     ref_element = finat.HsiehCloughTocher(ref_cell, 3)
#     ref_pts = finat.point_set.PointSet(ref_cell.make_points(2, 0, 4))

#     phys_cell = FIAT.ufc_simplex(2)
#     phys_cell.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))

#     mapping = MyMapping(ref_cell, phys_cell)
#     z = (0, 0)
#     finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mapping)[z]
#     finat_vals = evaluate([finat_vals_gem])[0].arr

#     phys_cell_FIAT = FIAT.HsiehCloughTocher(phys_cell, 3)
#     phys_points = phys_cell.make_points(2, 0, 4)
#     phys_vals = phys_cell_FIAT.tabulate(0, phys_points)[z]

#     assert np.allclose(finat_vals, phys_vals.T)


def test_reduced_hct():
    lat_ord = 3
    ref_cell = FIAT.ufc_simplex(2)
    ref_element = finat.ReducedHsiehCloughTocher(ref_cell, 3)
    # ref_element = finat.HsiehCloughTocher(ref_cell, 3)

    ref_pts = finat.point_set.PointSet(
        FIAT.reference_element.make_lattice(ref_cell.vertices, lat_ord))
    phys_cell = FIAT.ufc_simplex(2)

    phys_cell.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    # phys_cell.vertices = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    mapping = MyMapping(ref_cell, phys_cell)
    z = (0, 0)
    finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr

    finat_map_gem = ref_element.basis_transformation(mapping)
    print()
    print(evaluate([finat_map_gem])[0].arr.T)

    phys_cell_FIAT = FIAT.HsiehCloughTocher(phys_cell, 3, reduced=True)
    phys_points = FIAT.reference_element.make_lattice(phys_cell.vertices, lat_ord)
    phys_vals = phys_cell_FIAT.tabulate(0, phys_points)[z]

    diff = finat_vals.T - phys_vals
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if abs(diff[i, j] < 1.e-12):
                diff[i, j] = 0.0
    print(diff)
    assert np.allclose(finat_vals.T, phys_vals)
