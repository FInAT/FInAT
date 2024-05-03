import FIAT
import finat
import numpy as np
from gem.interpreter import evaluate

from fiat_mapping import MyMapping


def test_johnson_mercier():
    degree = 1
    ref_cell = FIAT.ufc_simplex(2)
    ref_el_finat = finat.JohnsonMercier(ref_cell, degree)
    ref_element = ref_el_finat._element
    ref_pts = ref_cell.make_points(2, 0, 3)
    ref_vals = ref_element.tabulate(0, ref_pts)[0, 0]

    phys_cell = FIAT.ufc_simplex(2)
    phys_cell.vertices = ((0.0, 0.0), (2.0, 0.1), (0.0, 1.0))
    phys_element = type(ref_element)(phys_cell, degree)

    phys_pts = phys_cell.make_points(2, 0, 3)
    phys_vals = phys_element.tabulate(0, phys_pts)[0, 0]

    # Piola map the reference elements
    J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                      phys_cell.vertices)
    detJ = np.linalg.det(J)

    ref_vals_piola = np.zeros(ref_vals.shape)
    for i in range(ref_vals.shape[0]):
        for k in range(ref_vals.shape[3]):
            ref_vals_piola[i, :, :, k] = \
                J @ ref_vals[i, :, :, k] @ J.T / detJ**2

    # Zany map the results
    mappng = MyMapping(ref_cell, phys_cell)
    Mgem = ref_el_finat.basis_transformation(mappng)
    M = evaluate([Mgem])[0].arr
    ref_vals_zany = np.zeros((15, 2, 2, len(phys_pts)))
    for k in range(ref_vals_zany.shape[3]):
        for ell1 in range(2):
            for ell2 in range(2):
                ref_vals_zany[:, ell1, ell2, k] = \
                    M @ ref_vals_piola[:, ell1, ell2, k]

    assert np.allclose(ref_vals_zany[:12], phys_vals[:12])
