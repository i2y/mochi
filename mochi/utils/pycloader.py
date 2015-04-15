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
        mochi_name = os.path.join(self.base_path, name + '.mochi')
        py_name = os.path.join(self.base_path, name + '_comp.pyc')
        pyc_compile_monkeypatch(mochi_name, py_name)
        return getattr(__import__(name), name)
