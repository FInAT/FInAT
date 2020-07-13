import pytest
import FIAT
import finat
import numpy as np
from gem.interpreter import evaluate
from fiat_mapping import MyMapping


def test_awnc():
    ref_cell = FIAT.ufc_simplex(2)
    ref_el_finat = finat.ArnoldWintherNC(ref_cell, 2)
    ref_element = ref_el_finat._element
    ref_pts = ref_cell.make_points(2, 0, 3)
    ref_vals = ref_element.tabulate(0, ref_pts)[0, 0]

    phys_cell = FIAT.ufc_simplex(2)
    phys_cell.vertices = ((0.0, 0.0), (2.0, 0.1), (0.0, 1.0))
    phys_element = ref_element.__class__(phys_cell, 2)

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
    mppng = MyMapping(ref_cell, phys_cell)
    Mgem = ref_el_finat.basis_transformation(mppng)
    M = evaluate([Mgem])[0].arr
    # print(M)
    # print(ref_vals_piola.shape)
    ref_vals_zany = np.zeros((15, 2, 2, len(phys_pts)))
    for k in range(ref_vals_zany.shape[3]):
        for ell1 in range(2):
            for ell2 in range(2):
                ref_vals_zany[:, ell1, ell2, k] = \
                    M @ ref_vals_piola[:, ell1, ell2, k]

    assert np.allclose(ref_vals_zany[:12], phys_vals[:12])


def test_awc():
    ref_cell = FIAT.ufc_simplex(2)
    ref_element = FIAT.ArnoldWinther(ref_cell, 3)
    ref_el_finat = finat.ArnoldWinther(ref_cell, 3)
    ref_pts = ref_cell.make_points(2, 0, 3)
    ref_vals = ref_element.tabulate(0, ref_pts)[0, 0]

    phys_cell = FIAT.ufc_simplex(2)
    pvs = np.array(((0.0, 0.0), (1.0, 0.1), (0.0, 2.0)))
    phys_cell.vertices = pvs * 0.5
    phys_element = FIAT.ArnoldWinther(phys_cell, 3)
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
    mppng = MyMapping(ref_cell, phys_cell)
    Mgem = ref_el_finat.basis_transformation(mppng)
    M = evaluate([Mgem])[0].arr
    # print(M)
    # print(ref_vals_piola.shape)
    ref_vals_zany = np.zeros((24, 2, 2, len(phys_pts)))
    for k in range(ref_vals_zany.shape[3]):
        for ell1 in range(2):
            for ell2 in range(2):
                ref_vals_zany[:, ell1, ell2, k] = \
                    M @ ref_vals_piola[:, ell1, ell2, k]

    # Q = FIAT.make_quadrature(phys_cell, 6)
    # vals = phys_element.tabulate(0, Q.pts)[(0, 0)]
    # print()
    # for i in (0, 9, 21):
    #     result = 0.0
    #     for k in range(len(Q.wts)):
    #         result += Q.wts[k] * (vals[i, :, :, k] ** 2).sum()
    #     print(np.sqrt(result))

    assert np.allclose(ref_vals_zany[:21], phys_vals[:21])
