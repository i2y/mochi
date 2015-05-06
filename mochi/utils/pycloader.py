"""Import a Python object made by compiling a Mochi file.
"""

import os

from mochi.core import init, pyc_compile_monkeypatch


def get_function(name, file_path):
    """Python function from Mochi.

    Compiles a Mochi file to Python bytecode and returns the
    imported function.
    """
    return getattr(get_module(name, file_path), name)


def get_module(full_name, file_path):
    """Python function from Mochi.

    Compiles a Mochi file to Python bytecode and returns the
    Python module.
    """
    name = full_name.split('.')[-1]
    base_path = os.path.dirname(file_path)
    mochi_name = os.path.join(base_path, name + '.mochi')
    py_name = os.path.join(base_path, name + '.pyc')
    pyc_compile_monkeypatch(mochi_name, py_name)
    return __import__(full_name)

init()
