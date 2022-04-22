import FIAT
import finat
import numpy as np
from gem.interpreter import evaluate
from fiat_mapping import MyMapping
from finat.wu_xu_fiat import WuXuH3NC as wxfiat
#from finat.wu_xu_fiat import WuXuRobustH3NC as wxfiat
from finat.wu_xu_fiat import compact_transform

def test_wuxu():
    ref_cell = FIAT.ufc_simplex(2)
    ref_element = finat.WuXuH3NC(ref_cell, 4)
    ref_pts = finat.point_set.PointSet(ref_cell.make_points(2, 0, 4))

    phys_cell = FIAT.ufc_simplex(2)
    phys_cell.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))

    mapping = MyMapping(ref_cell, phys_cell)
    z = (0, 0)
    finat_vals_gem = ref_element.basis_evaluation(0, ref_pts, coordinate_mapping=mapping)[z]
    finat_vals = evaluate([finat_vals_gem])[0].arr

    phys_cell_FIAT = wxfiat(phys_cell)
    phys_points = phys_cell.make_points(2, 0, 4)
    phys_vals = phys_cell_FIAT.tabulate(0, phys_points)[z]

    #assert np.allclose(np.transpose(ref_element.basis_transformation(mapping).array), compact_transform(phys_cell, ref_cell))

    print(phys_vals / finat_vals.T)

    assert np.allclose(finat_vals, phys_vals.T)
