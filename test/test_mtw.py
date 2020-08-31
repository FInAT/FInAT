import FIAT
import finat
import numpy as np
from gem.interpreter import evaluate
from fiat_mapping import MyMapping


def test_mtw():
    ref_cell = FIAT.ufc_simplex(2)
    ref_el_finat = finat.MardalTaiWinther(ref_cell, 3)
    ref_element = ref_el_finat._element
    ref_pts = ref_cell.make_points(2, 0, 3)
    ref_vals = ref_element.tabulate(0, ref_pts)[0, 0]

    phys_cell = FIAT.ufc_simplex(2)
    phys_cell.vertices = ((0.0, 0.0), (2.0, 0.1), (0.0, 1.0))
    phys_element = ref_element.__class__(phys_cell, 3)

    phys_pts = phys_cell.make_points(2, 0, 3)
    phys_vals = phys_element.tabulate(0, phys_pts)[0, 0]

    # Piola map the reference elements
    J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                      phys_cell.vertices)
    detJ = np.linalg.det(J)

    ref_vals_piola = np.zeros(ref_vals.shape)
    for i in range(ref_vals.shape[0]):
        for k in range(ref_vals.shape[2]):
            ref_vals_piola[i, :, k] = \
                J @ ref_vals[i, :, k] / detJ

    # Zany map the results
    mappng = MyMapping(ref_cell, phys_cell)
    Mgem = ref_el_finat.basis_transformation(mappng)
    M = evaluate([Mgem])[0].arr
    ref_vals_zany = np.zeros((9, 2, len(phys_pts)))
    for k in range(ref_vals_zany.shape[2]):
        for ell in range(2):
            ref_vals_zany[:, ell, k] = \
                M @ ref_vals_piola[:, ell, k]

    assert np.allclose(ref_vals_zany[:9], phys_vals[:9])
