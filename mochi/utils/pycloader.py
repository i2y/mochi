"""Import a Python object made by compiling a Mochi file.
"""

import os

from mochi.core import pyc_compile_monkeypatch


class PycLoader(object):
    """Loading Mochi functions into Python.
    """

    def __init__(self, file_path):
        self.base_path = os.path.dirname(file_path)

    def get_function(self, name):
        """Python function from Mochi.

        Compiles a Mochi file to Python bytecode and returns the
        imported function.
        """
        return getattr(self.get_module(name), name)

    def get_module(self, name):
        """Python function from Mochi.

        Compiles a Mochi file to Python bytecode and returns the
        Python module.
        """
        mochi_name = os.path.join(self.base_path, name + '.mochi')
        py_name = os.path.join(self.base_path, name + '.pyc')
        pyc_compile_monkeypatch(mochi_name, py_name)
        return __import__(name)
