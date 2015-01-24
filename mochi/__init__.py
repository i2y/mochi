__author__ = 'Yasushi Itoh'
__version__ = '0.1.6'
__license__ = 'MIT License'


import platform
import sys

IS_PYTHON_34 = sys.version_info >= (3, 4)
IS_PYPY = platform.python_implementation() == 'PyPy'
