FInAT is not a tabulator
========================

FInAT is an attempt to provide a more abstract, smarter library of
finite elements. FInAT will supply symbolic information about the
properties of individual finite elements, and will provide abstract
algorithms for their evaluation. This enables FInAT to provide smart
algorithms which exploit the internal structure of the element
basis. The result will be a separation of concerns in which FInAT
provides a single source of truth for information about the symbolic
and numerical properties of finite elements. Symbolic form languages
and form compilers can then exploit this information in an
element-independent manner.

