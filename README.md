# FInAT: a smarter library of finite elements

Unlike [FIAT](https://github.com/FEniCS/fiat "FInite element Automatic
Tabulator"), “**F**InAT **I**s **n**ot **A** **T**abulator.”  Instead,
it provides symbolic expressions for the evaluation of finite
elements.  Thus FInAT can express the structure that is intrinsic to
some finite elements, which a form compiler can exploit to
automatically:

- sum factorise continuous, discontinuous, *H*(div) and *H*(curl)
  conforming elements on cuboid cells;
- optimise evaluation at collocated quadrature points when
  under-integration is requested; and
- optimise evaluation on vector and tensor elements.

Furthermore, FInAT is equipped to provide symbolic expressions for
basis transformations when the caller implements an interface capable
of providing (symbolic expressions for) the required geometric
quantities (such as Jacobians, normals, or tangents), which
facilitates the implementation of finite elements such as Hermite,
Morley, Bell, and Argyris within an automatic code generation
framework.

The goal of FInAT is to be a “single source of truth” for finite
elements. This includes not only basis function evaluation, like FIAT,
but also structural and algorithmic considerations as well as
reference element transformations.  Symbolic expression languages and
form compilers can then exploit this information in an
element-independent manner.

FInAT is integrated with
[TSFC](https://github.com/firedrakeproject/tsfc "Two-Stage Form
Compiler") and is a component of
[Firedrake](https://firedrakeproject.org/), “an automated system for
the portable solution of partial differential equations using the
finite element method.”  To facilitate the exchange of symbolic
expressions between the element library on the one hand, and the form
compiler on the other, they need to agree on a common language.  Where
FIAT communicated with the form compiler through numerical arrays,
FInAT communicates with the form compiler by exchanging GEM
expressions.  GEM is the intermediate language used in both TSFC and
FInAT to describe tensor algebra.

## References

1. Miklós Homolya, Robert C. Kirby, and David A. Ham (2017).
   “Exposing and exploiting structure: optimal code generation for
   high-order finite element methods.” _arXiv preprint
   arXiv:1711.02473._
1. Robert C. Kirby, and Lawrence Mitchell (2019). “Code generation for
   generally mapped finite elements.” _ACM Transactions on
   Mathematical Software (TOMS)_, 45(4), pp. 1-23.

## License

All files in this repository are available under the MIT license, see
the LICENSE file for details.
