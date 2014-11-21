import pytest
import finat
from finat.ast import FInATSyntaxError, Recipe, IndexSum, Let, Wave
import pymbolic.primitives as p


@pytest.fixture
def i():
    return finat.indices.DimensionIndex(10)


def test_invalid_binding(i):
    e = Recipe(((), (i,), ()), IndexSum((i,), 1))
    with pytest.raises(FInATSyntaxError):
        finat.interpreter.evaluate(e)


def test_index_sum(i):
    e = Recipe(((), (), ()), IndexSum((i,), 1))
    assert finat.interpreter.evaluate(e) == 10


def test_nested_sum():
    i1 = finat.indices.DimensionIndex(4)
    i2 = finat.indices.DimensionIndex(i1)

    e = Recipe(((), (), ()), IndexSum((i1, i2), 1))
    assert finat.interpreter.evaluate(e) == 6


def test_let(i):
    v = p.Variable("v")
    e = Recipe(((), (), ()), Let(((v, 1),), IndexSum((i,), v)))
    assert finat.interpreter.evaluate(e) == 10


def test_wave(i):
    v = p.Variable("v")
    e = Recipe(((), (), ()), IndexSum((i,), Wave(v, i, 0, v + 1, v)))
    assert finat.interpreter.evaluate(e) == 45


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
