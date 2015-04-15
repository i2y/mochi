import os

from mochi.utils.pycloader import PycLoader

loader = PycLoader(__file__)
factorial = loader.get_function('factorial')

def test_factorial_1():
    assert factorial(1) == 1

def test_factorial_2():
    assert factorial(2) == 2



