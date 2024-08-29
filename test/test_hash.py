import os
import sys
import subprocess
import textwrap

import ufl
import finat.ufl


def test_same_hash():
    """ The same element created twice should have the same hash.
    """
    cg = finat.ufl.finiteelement.FiniteElement("Lagrange", ufl.cell.Cell("triangle"), 1)
    same_cg = finat.ufl.finiteelement.FiniteElement("Lagrange", ufl.cell.Cell("triangle"), 1)
    assert hash(cg) == hash(same_cg)


def test_different_hash():
    """ Two different elements should have different hashes.
    """
    cg = finat.ufl.finiteelement.FiniteElement("Lagrange", ufl.cell.Cell("triangle"), 1)
    dg = finat.ufl.finiteelement.FiniteElement("DG", ufl.cell.Cell("triangle"), 2)
    assert hash(cg) != hash(dg)


def test_variant_hashes_different():
    """ Different variants of the same element should have different hashes.
    """
    dg = finat.ufl.finiteelement.FiniteElement("DG", ufl.cell.Cell("triangle"), 2)
    dg_gll = finat.ufl.finiteelement.FiniteElement("DG", ufl.cell.Cell("triangle"), 2, variant="gll")
    assert hash(dg) != hash(dg_gll)


def test_persistent_hash(tmp_path):
    """ Hashes should be the same across Python invocations.
    """
    filename = "print_hash.py"
    code = textwrap.dedent("""\
        import ufl
        import finat.ufl

        dg = finat.ufl.finiteelement.FiniteElement("RT", ufl.cell.Cell("triangle"), 1)
        print(hash(dg))
        """)
    filepath = tmp_path.joinpath(filename)
    with open(filepath, "w") as fh:
        fh.write(code)

    output1 = subprocess.run([sys.executable, filepath], capture_output=True)
    assert output1.returncode == os.EX_OK
    output2 = subprocess.run([sys.executable, filepath], capture_output=True)
    assert output2.returncode == os.EX_OK
    assert output1.stdout == output2.stdout
