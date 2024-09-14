import FIAT
import finat
import numpy as np
import pytest
from gem.interpreter import evaluate

from fiat_mapping import MyMapping


def make_unisolvent_points(element):
    degree = element.degree()
    ref_complex = element.get_reference_complex()
    sd = ref_complex.get_spatial_dimension()
    top = ref_complex.get_topology()
    pts = []
    for cell in top[sd]:
        pts.extend(ref_complex.make_points(sd, cell, degree+sd+1))
    return pts


@pytest.mark.parametrize('phys_verts',
                         [((0.0, 0.0), (2.0, 0.1), (0.0, 1.0)),
                          ((0, 0, 0), (1., 0.1, -0.37),
                           (0.01, 0.987, -.23),
                           (-0.1, -0.2, 1.38))])
def test_johnson_mercier(phys_verts):
    degree = 1
    variant = None
    sd = len(phys_verts) - 1
    ref_cell = FIAT.ufc_simplex(sd)
    ref_el_finat = finat.JohnsonMercier(ref_cell, degree, variant=variant)
    indices = ref_el_finat._indices

    ref_element = ref_el_finat._element
    shape = ref_element.value_shape()
    ref_pts = make_unisolvent_points(ref_element)
    ref_vals = ref_element.tabulate(0, ref_pts)[(0,)*sd]

    phys_cell = FIAT.ufc_simplex(sd)
    phys_cell.vertices = phys_verts
    phys_element = type(ref_element)(phys_cell, degree, variant=variant)

    phys_pts = make_unisolvent_points(phys_element)
    phys_vals = phys_element.tabulate(0, phys_pts)[(0,)*sd]

    # Piola map the reference elements
    J, b = FIAT.reference_element.make_affine_mapping(ref_cell.vertices,
                                                      phys_cell.vertices)
    detJ = np.linalg.det(J)
    K = J / detJ

    ref_vals_piola = np.zeros(ref_vals.shape)
    for i in range(ref_vals.shape[0]):
        for k in range(ref_vals.shape[-1]):
            ref_vals_piola[i, ..., k] = K @ ref_vals[i, ..., k] @ K.T

    dofs = ref_el_finat.entity_dofs()
    num_bfs = phys_element.space_dimension()
    num_dofs = ref_el_finat.space_dimension()
    num_facet_dofs = num_dofs - sum(len(dofs[sd][entity]) for entity in dofs[sd])

    # Zany map the results
    mappng = MyMapping(ref_cell, phys_cell)
    Mgem = ref_el_finat.basis_transformation(mappng)
    M = evaluate([Mgem])[0].arr
    shp = (num_dofs, *shape, len(phys_pts))
    ref_vals_zany = (M @ ref_vals_piola.reshape(num_bfs, -1)).reshape(shp)

    # Solve for the basis transformation and compare results
    Phi = ref_vals_piola.reshape(num_bfs, -1)
    phi = phys_vals.reshape(num_bfs, -1)
    Mh = np.linalg.solve(Phi @ Phi.T, Phi @ phi.T).T
    Mh[abs(Mh) < 1E-10] = 0.0
    assert np.allclose(M[:num_facet_dofs], Mh[indices][:num_facet_dofs])

    assert np.allclose(ref_vals_zany[:num_facet_dofs], phys_vals[indices][:num_facet_dofs])
