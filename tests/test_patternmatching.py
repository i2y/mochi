import os

from pyrsistent import pvector
from mochi.utils.pycloader import PycLoader

loader = PycLoader(__file__)
factorial = loader.get_function('factorial')
fizzbuzz = loader.get_function('fizzbuzz')
patterns = loader.get_module('patterns')


def test_factorial_1():
    assert factorial(1) == 1


def test_factorial_2():
    assert factorial(2) == 2


def test_fizzbuzz():
    res = [1, 2, 'fizz', 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz',
           13, 14, 'fizzbuzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23,
           'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizzbuzz']
    assert [fizzbuzz(n) for n in range(1, 31)] == res


def test_list_seq():
    assert patterns.list_seq() == 3


def test_match_rest():
    assert patterns.match_rest() == pvector([2, 3])


def test_list_seq_auto():
    assert patterns.list_seq() == patterns.result_list_seq()
