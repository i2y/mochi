import pytest

from mochi.utils.pycloader import get_module

from conftest import auto_test


macro = get_module('macro', file_path=__file__)


@pytest.mark.parametrize("func,result", auto_test(macro))
def test_patterns(func, result):
    assert func() == result()