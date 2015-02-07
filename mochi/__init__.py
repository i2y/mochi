__author__ = 'Yasushi Itoh'
__version__ = '0.1.8'
__license__ = 'MIT License'


import platform
import sys

GE_PYTHON_34 = sys.version_info >= (3, 4)
GE_PYTHON_33 = sys.version_info >= (3, 3)
IS_PYPY = platform.python_implementation() == 'PyPy'
