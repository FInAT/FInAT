import FIAT
import finat
import numpy as np
import pytest
from gem.interpreter import evaluate

from fiat_mapping import MyMapping


@pytest.mark.parametrize('phys_verts',
                         [((0.0, 0.0), (2.0, 0.1), (0.0, 1.0)),
                          ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))])
def test_johnson_mercier(phys_verts):
    degree = 1
    sd = len(phys_verts) - 1
    z = tuple(0 for _ in range(sd))
    ref_cell = FIAT.ufc_simplex(sd)
    ref_el_finat = finat.JohnsonMercier(ref_cell, degree)
    ref_element = ref_el_finat._element
    ref_pts = ref_cell.make_points(sd, 0, 1+sd)
    ref_vals = ref_element.tabulate(0, ref_pts)[z]

    phys_cell = FIAT.ufc_simplex(sd)
    phys_cell.vertices = phys_verts
    phys_element = type(ref_element)(phys_cell, degree)

    phys_pts = phys_cell.make_points(sd, 0, 1+sd)
    phys_vals = phys_element.tabulate(0, phys_pts)[z]

    # Piola map the reference elements
    J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                      phys_cell.vertices)
    detJ = np.linalg.det(J)

    ref_vals_piola = np.zeros(ref_vals.shape)
    for i in range(ref_vals.shape[0]):
        for k in range(ref_vals.shape[3]):
            ref_vals_piola[i, :, :, k] = \
                J @ ref_vals[i, :, :, k] @ J.T / detJ**2

    num_bfs = phys_element.space_dimension()
    num_facet_bfs = (sd + 1) * len(phys_element.dual.entity_ids[sd-1][0])

    # Zany map the results
    mappng = MyMapping(ref_cell, phys_cell)
    Mgem = ref_el_finat.basis_transformation(mappng)
    M = evaluate([Mgem])[0].arr
    ref_vals_zany = np.zeros((num_bfs, sd, sd, len(phys_pts)))
    for k in range(ref_vals_zany.shape[3]):
        for ell1 in range(sd):
            for ell2 in range(sd):
                ref_vals_zany[:, ell1, ell2, k] = \
                    M @ ref_vals_piola[:, ell1, ell2, k]

    print(np.diag(M))
    # print((ref_vals_zany[:num_facet_bfs] - phys_vals[:num_facet_bfs]).T)
    assert np.allclose(ref_vals_zany[:num_facet_bfs], phys_vals[:num_facet_bfs])
