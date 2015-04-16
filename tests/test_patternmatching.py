import os
from functools import partial

import pytest

from mochi.utils.pycloader import get_function, get_module

from conftest import auto_test

get_module = partial(get_module, file_path=__file__)
get_function = partial(get_function, file_path=__file__)

factorial = get_function('factorial')
fizzbuzz = get_function('fizzbuzz')
patterns = get_module('patterns')


def test_factorial_1():
    assert factorial(1) == 1


def test_factorial_2():
    assert factorial(2) == 2


def test_fizzbuzz():
    res = [1, 2, 'fizz', 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz',
           13, 14, 'fizzbuzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23,
           'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizzbuzz']
    assert [fizzbuzz(n) for n in range(1, 31)] == res


@pytest.mark.parametrize("func,result", auto_test(patterns))
def test_patterns(func, result):
    assert func() == result()

