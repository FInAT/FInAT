import FIAT
import gem
import numpy as np
from finat.physically_mapped import PhysicalGeometry


class MyMapping(PhysicalGeometry):
    def __init__(self, ref_cell, phys_cell):
        self.ref_cell = ref_cell
        self.phys_cell = phys_cell

        self.A, self.b = FIAT.reference_element.make_affine_mapping(
            self.ref_cell.vertices,
            self.phys_cell.vertices)

    def cell_size(self):
        # Firedrake interprets this as 2x the circumradius
        # cs = (np.prod([self.phys_cell.volume_of_subcomplex(1, i)
        #                for i in range(3)])
        #       / 2.0 / self.phys_cell.volume())
        # return np.asarray([cs for _ in range(3)])
        # Currently, just return 1 so we can compare FIAT dofs
        # to transformed dofs.

        return np.ones((3,))

    def detJ_at(self, point):
        return gem.Literal(np.linalg.det(self.A))

    def jacobian_at(self, point):
        return gem.Literal(self.A)

    def normalized_reference_edge_tangents(self):
        return gem.Literal(np.asarray([self.ref_cell.compute_normalized_edge_tangent(i) for i in range(3)]))

    def reference_normals(self):
        return gem.Literal(
            np.asarray([self.ref_cell.compute_normal(i)
                        for i in range(3)]))

    def physical_normals(self):
        return gem.Literal(
            np.asarray([self.phys_cell.compute_normal(i)
                        for i in range(3)]))

    def physical_tangents(self):
        return gem.Literal(
            np.asarray([self.phys_cell.compute_normalized_edge_tangent(i)
                        for i in range(3)]))

    def physical_edge_lengths(self):
        return gem.Literal(
            np.asarray([self.phys_cell.volume_of_subcomplex(1, i)
                        for i in range(3)]))

    def physical_points(self, ps, entity=None):
        prefs = ps.points
        A, b = self.A, self.b
        return gem.Literal(np.asarray([A @ x + b for x in prefs]))

    def physical_vertices(self):
        return gem.Literal(self.phys_cell.verts)


class FiredrakeMapping(MyMapping):

    def cell_size(self):
        # Firedrake interprets this as 2x the circumradius
        cs = (np.prod([self.phys_cell.volume_of_subcomplex(1, i)
                       for i in range(3)])
              / 2.0 / self.phys_cell.volume())
        return np.asarray([cs for _ in range(3)])
