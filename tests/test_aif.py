import pytest

from mochi.utils.pycloader import get_module

from conftest import auto_test


aif = get_module('aif', file_path=__file__)


@pytest.mark.parametrize("func,result", auto_test(aif))
def test_patterns(func, result):
    assert func() == result()