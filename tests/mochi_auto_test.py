"""Automatic testing of all functions in `mochi_test_*.mochi files.
"""

import itertools as it
import pathlib

import pytest

from mochi.utils.pycloader import get_module

from conftest import auto_test


def find_mod_names(file_path=__file__):
    """Find Mochi module names that start with 'mochi_test_'.

    The returned names are without the extension '.mochi'.
    TODO: Needs to recurse in `test_` sub directories.
    """
    directory = pathlib.Path(file_path).absolute().parent
    # pylint: disable=no-member
    return [f_name.stem for f_name in directory.iterdir() if
            f_name.stem.startswith('mochi_test_') and
            f_name.suffix == '.mochi']


def make_paramters(module_names, file_path=__file__):
    """Create parameters for py.test.generated tests.

    Creates a generator with pairs of functions.
    The first function is to be tested and the second functions returns
    th expected result of the first function.
    Converted into a list it would like this:
    [<function func_1>, <function result_func_1>,
     <function func_2>, <function result_func_2>,
     ...
     <function func_n>, <function result_func_n>,]
    """
    modules = [get_module(m_name, file_path) for m_name in module_names]
    params = (auto_test(module) for module in modules)
    return it.chain(*params)  # pylint: disable=star-args


@pytest.mark.parametrize("func,result", make_paramters(find_mod_names()))
def test_patterns(func, result):
    """Test that func returns same values as result.

    This test is parameterized and will be called by py.test for all
    (func, result) pairs in `parameters.
    """
    print('Error in {}.{}.'.format(func.__module__, func.__qualname__))
    assert func() == result()
