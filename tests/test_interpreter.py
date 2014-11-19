import pytest
import FIAT
import finat
import pymbolic.primitives as p
import numpy as np
from finat.ast import FInATSyntaxError

@pytest.fixture
def i():
    return finat.indices.DimensionIndex(10)


def test_invalid_binding(i):
    e = finat.ast.Recipe(((), (i,), ()), finat.ast.IndexSum((i,), 1))
    with pytest.raises(FInATSyntaxError):
        finat.interpreter.evaluate(e)

def test_invalid_binding(i):
    e = finat.ast.Recipe(((), (), ()), finat.ast.IndexSum((i,), 1))
    assert finat.interpreter.evaluate(e) == 10

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
