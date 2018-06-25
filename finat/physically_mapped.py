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
except ImportError:
    Citations = None


class PhysicallyMappedElement(metaclass=ABCMeta):
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
            i = gem.Index()
            j = gem.Index()
            val = gem.ComponentTensor(
                gem.IndexSum(gem.Product(gem.Indexed(M, (i, j)),
                                         gem.Indexed(table, (j,))),
                             (j,)),
                (i,))
            # Eliminate zeros
            return gem.optimise.aggressive_unroll(val)

        result = super().basis_evaluation(order, ps, entity=entity)

        return {alpha: matvec(table)
                for alpha, table in result.items()}

    def point_evaluation(self, order, refcoords, entity=None):
        raise NotImplementedError("TODO: not yet thought about it")
