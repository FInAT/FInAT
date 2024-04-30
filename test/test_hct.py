import FIAT
import finat
import numpy as np
from gem.interpreter import evaluate

from fiat_mapping import MyMapping


def test_hct():
    degree = 3
    ref_cell = FIAT.ufc_simplex(2)
    ref_pts = finat.point_set.PointSet(FIAT.reference_element.make_lattice(ref_cell.vertices, degree))
    ref_element = finat.HsiehCloughTocher(ref_cell, degree, avg=True)

    phys_cell = FIAT.ufc_simplex(2)
    phys_cell.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))

    mapping = MyMapping(ref_cell, phys_cell)
    z = (0, 0)
    finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr

    phys_cell_FIAT = FIAT.HsiehCloughTocher(phys_cell, degree)
    phys_points = FIAT.reference_element.make_lattice(phys_cell.vertices, degree)
    phys_vals = phys_cell_FIAT.tabulate(0, phys_points)[z]

    assert np.allclose(finat_vals.T, phys_vals)


def test_reduced_hct():
    degree = 3
    ref_cell = FIAT.ufc_simplex(2)
    ref_pts = finat.point_set.PointSet(FIAT.reference_element.make_lattice(ref_cell.vertices, degree))
    ref_element = finat.ReducedHsiehCloughTocher(ref_cell, degree)

    phys_cell = FIAT.ufc_simplex(2)
    phys_cell.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))

    mapping = MyMapping(ref_cell, phys_cell)
    z = (0, 0)
    finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr

    phys_cell_FIAT = FIAT.HsiehCloughTocher(phys_cell, degree, reduced=True)
    phys_points = FIAT.reference_element.make_lattice(phys_cell.vertices, degree)
    phys_vals = phys_cell_FIAT.tabulate(0, phys_points)[z]

    numbf = ref_element.space_dimension()
    assert np.allclose(finat_vals.T, phys_vals[:numbf])
