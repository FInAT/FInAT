import pytest
import FIAT
import finat
import gem
import numpy as np
from finat.physically_mapped import PhysicalGeometry


class MyMapping(PhysicalGeometry):
    def __init__(self, cell, verts):
        # cell is reference cell, verts is physical vertices
        self.verts = np.asarray(verts)
        self.cell = cell

    def cell_size(self):
        raise NotImplementedError

    def jacobian_at(self, point):
        raise NotImplementedError

    def reference_normals(self):
        raise NotImplementedError

    def physical_normals(self):
        raise NotImplementedError

    def physical_tangents(self):
        raise NotImplementedError

    def physical_edge_lengths(self):
        raise NotImplementedError

    def physical_points(self, ps, entity=None):
        assert entity is None
        prefs = ps.points
        pvs = self.verts
        pps = np.zeros(prefs.shape, dtype=float)
        for i in range(pps.shape[0]):
            pps[i, :] = (pvs[0, :] * (1-prefs[i, 0]) * (1-prefs[i, 1])
                         + pvs[1, :] * (1-prefs[i, 0]) * prefs[i, 1]
                         + pvs[2, :] * prefs[i, 0] * (1-prefs[i, 1])
                         + pvs[3, :] * prefs[i, 0] * prefs[i, 1])
        return gem.Literal(pps)

    def physical_vertices(self):
        return gem.Literal(self.verts)


def get_pts(cell, deg):
    assert cell.shape == FIAT.reference_element.QUADRILATERAL
    L = cell.construct_subelement(1)
    vs = np.asarray(cell.vertices)
    pts = [pt for pt in cell.vertices]
    Lpts = FIAT.reference_element.make_lattice(L.vertices, deg, 1)
    for e in cell.topology[1]:
        Fmap = cell.get_entity_transform(1, e)
        epts = [tuple(Fmap(pt)) for pt in Lpts]
        pts.extend(epts)
    if deg > 3:
        dx0 = (vs[1, :] - vs[0, :]) / (deg-2)
        dx1 = (vs[2, :] - vs[0, :]) / (deg-2)

        internal_nodes = [tuple(vs[0, :] + dx0 * i + dx1 * j)
                          for i in range(1, deg-2)
                          for j in range(1, deg-1-i)]
        pts.extend(internal_nodes)
    return pts


@pytest.mark.parametrize('degree', [1, 2, 3, 4])
def test_kronecker(degree):
    cell = FIAT.ufc_cell("quadrilateral")
    element = finat.DirectSerendipity(cell, degree)
    pts = finat.point_set.PointSet(get_pts(cell, degree))
    vrts = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.1, 1.1), (0.95, 1.01)))
    mppng = MyMapping(cell, vrts)
    z = tuple([0] * cell.get_spatial_dimension())
    vals = element.basis_evaluation(0, pts, coordinate_mapping=mppng)[z]
    from gem.interpreter import evaluate
    numvals = evaluate([vals])[0].arr
    assert np.allclose(numvals, np.eye(*numvals.shape))


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
