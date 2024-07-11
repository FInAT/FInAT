import FIAT
import finat
import numpy as np
import pytest
from gem.interpreter import evaluate

from fiat_mapping import FiredrakeMapping


@pytest.mark.parametrize("element, degree, variant", [
                         (finat.Hermite, 3, None),
                         (finat.ReducedHsiehCloughTocher, 3, None),
                         (finat.HsiehCloughTocher, 3, None),
                         (finat.HsiehCloughTocher, 4, None),
                         (finat.Bell, 5, None),
                         (finat.Argyris, 5, "point"),
                         (finat.Argyris, 5, None),
                         (finat.Argyris, 6, None),
                         ])
def test_mass_scaling(element, degree, variant):
    sd = 2
    ref_cell = FIAT.ufc_simplex(sd)
    if variant is not None:
        ref_element = element(ref_cell, degree, variant=variant)
    else:
        ref_element = element(ref_cell, degree)

    Q = finat.quadrature.make_quadrature(ref_cell, 2*degree)
    qpts = Q.point_set
    qwts = Q.weights

    kappa = []
    for k in range(3):
        h = 2 ** -k
        phys_cell = FIAT.ufc_simplex(2)
        new_verts = h * np.array(phys_cell.get_vertices())
        phys_cell.vertices = tuple(map(tuple, new_verts))
        mapping = FiredrakeMapping(ref_cell, phys_cell)
        J_gem = mapping.jacobian_at(ref_cell.make_points(sd, 0, sd+1)[0])
        J = evaluate([J_gem])[0].arr

        z = (0,) * ref_element.cell.get_spatial_dimension()
        finat_vals_gem = ref_element.basis_evaluation(0, qpts, coordinate_mapping=mapping)[z]
        phis = evaluate([finat_vals_gem])[0].arr.T

        M = np.dot(np.multiply(phis, qwts * abs(np.linalg.det(J))), phis.T)
        kappa.append(np.linalg.cond(M))

    kappa = np.array(kappa)
    ratio = kappa[1:] / kappa[:-1]
    assert np.allclose(ratio, 1, atol=0.1)
