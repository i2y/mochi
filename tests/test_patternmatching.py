import os

from mochi.utils.pycloader import get_function

factorial = get_function('factorial', __file__)

def test_factorial_1():
    assert factorial(1) == 1

def test_factorial_2():
    assert factorial(2) == 2



