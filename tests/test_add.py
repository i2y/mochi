# file test_add.py in dir tests

from mochi.utils.pycloader import get_module


mod_add = get_module('add', file_path=__file__)


def test_add():
    assert mod_add.add(2, 2) == 4


def test_add10():
    assert mod_add.add10(2) == 12
