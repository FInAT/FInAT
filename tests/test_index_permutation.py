import pytest
import finat


@pytest.mark.xfail
def test_index_permuation():

    i = finat.DimensionIndex(3)
    j = finat.DimensionIndex(2)

    recipe = finat.Recipe(((i, j), (), ()),
                          finat.ForAll((j,),
                                       finat.ForAll((i,), 1.)))

    out = finat.interpreter.evaluate(recipe)

    assert out.shape == (3, 2)
