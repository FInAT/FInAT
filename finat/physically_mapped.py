import gem
from abc import ABCMeta, abstractmethod

try:
    from firedrake_citations import Citations
    Citations().add("Kirby2018zany", """
@Article{Kirby2018zany,
  author =       {Robert C. Kirby},
  title =        {A general approach to transforming finite elements},
  journal =      {SMAI Journal of Computational Mathematics},
  year =         2018,
  volume =       4,
  pages =        {197-224},
  doi =          {10.5802/smai-jcm.33},
  archiveprefix ={arXiv},
  eprint =       {1706.09017},
  primaryclass = {math.NA}
}
""")
    Citations().add("Argyris1968", """
@Article{Argyris1968,
  author =       {J. H. Argyris and I. Fried and D. W. Scharpf},
  title =        {{The TUBA family of plate elements for the matrix
                  displacement method}},
  journal =      {The Aeronautical Journal},
  year =         1968,
  volume =       72,
  pages =        {701-709},
  doi =          {10.1017/S000192400008489X}
}
""")
    Citations().add("Bell1969", """
@Article{Bell1969,
  author =       {Kolbein Bell},
  title =        {A refined triangular plate bending finite element},
  journal =      {International Journal for Numerical Methods in
                  Engineering},
  year =         1969,
  volume =       1,
  number =       1,
  pages =        {101-122},
  doi =          {10.1002/nme.1620010108}
}
""")
    Citations().add("Ciarlet1972", r"""
@Article{Ciarlet1972,
  author =       {P. G. Ciarlet and P. A. Raviart},
  title =        {{General Lagrange and Hermite interpolation in
                  $\mathbb{R}^n$ with applications to finite element
                  methods}},
  journal =      {Archive for Rational Mechanics and Analysis},
  year =         1972,
  volume =       46,
  number =       3,
  pages =        {177-199},
  doi =          {10.1007/BF0025245}
}
""")
    Citations().add("Morley1971", """
@Article{Morley1971,
  author =       {L. S. D. Morley},
  title =        {The constant-moment plate-bending element},
  journal =      {The Journal of Strain Analysis for Engineering
                  Design},
  year =         1971,
  volume =       6,
  number =       1,
  pages =        {20-24},
  doi =          {10.1243/03093247V061020}
}
""")
except ImportError:
    Citations = None


class NeedsCoordinateMappingElement(metaclass=ABCMeta):
    """Abstract class for elements that require physical information
    either to map or construct their basis functions."""
    pass
    

class PhysicallyMappedElement(NeedsCoordinateMappingElement):
    """A mixin that applies a "physical" transformation to tabulated
    basis functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if Citations is not None:
            Citations().register("Kirby2018zany")

    @abstractmethod
    def basis_transformation(self, coordinate_mapping):
        """Transformation matrix for the basis functions.

        :arg coordinate_mapping: Object providing physical geometry."""
        pass

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        assert coordinate_mapping is not None

        M = self.basis_transformation(coordinate_mapping)

        def matvec(table):
            i, j = gem.indices(2)
            val = gem.ComponentTensor(gem.IndexSum(M[i, j]*table[j], (j,)), (i,))
            # Eliminate zeros
            return gem.optimise.aggressive_unroll(val)

        result = super().basis_evaluation(order, ps, entity=entity)

        return {alpha: matvec(table)
                for alpha, table in result.items()}

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError("TODO: not yet thought about it")


class DirectlyDefinedElement(NeedsCoordinateMappingElement):
    """Base class for directly defined elements such as direct
    serendipity that bypass a coordinate mapping."""
    pass
    

class PhysicalGeometry(metaclass=ABCMeta):

    @abstractmethod
    def cell_size(self):
        """The cell size at each vertex.

        :returns: A GEM expression for the cell size, shape (nvertex, ).
        """

    @abstractmethod
    def jacobian_at(self, point):
        """The jacobian of the physical coordinates at a point.

        :arg point: The point in reference space to evaluate the Jacobian.
        :returns: A GEM expression for the Jacobian, shape (gdim, tdim).
        """

    @abstractmethod
    def reference_normals(self):
        """The (unit) reference cell normals for each facet.

        :returns: A GEM expression for the normal to each
           facet (numbered according to FIAT conventions), shape
           (nfacet, tdim).
        """

    @abstractmethod
    def physical_normals(self):
        """The (unit) physical cell normals for each facet.

        :returns: A GEM expression for the normal to each
           facet (numbered according to FIAT conventions).  These are
           all computed by a clockwise rotation of the physical
           tangents, shape (nfacet, gdim).
        """

    @abstractmethod
    def physical_tangents(self):
        """The (unit) physical cell tangents on each facet.

        :returns: A GEM expression for the tangent to each
           facet (numbered according to FIAT conventions).  These
           always point from low to high numbered local vertex, shape
           (nfacet, gdim).
        """

    @abstractmethod
    def physical_edge_lengths(self):
        """The length of each edge of the physical cell.

        :returns: A GEM expression for the length of each
           edge (numbered according to FIAT conventions), shape
           (nfacet, ).
        """

    @abstractmethod
    def physical_points(self, pt_set):
        """Maps reference element points to GEM for the physical coordinates
        
        :returns a GEM expression for the physical locations of the points
        """

    @abstractmethod
    def physical_vertices(self):
        """Physical locations of the cell vertices.

        :returns a GEM expression for the physical vertices."""

