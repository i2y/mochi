"""Generate Python source files from Mochi files and check for valid syntax.
"""

import glob
import os
import py_compile

from mochi.core.main import make_py_source_file, MONKEY_PATCH_ENV
from mochi.utils.path_helper import TempDir


def make_target_file_name(file_name, dst_path, ext):
    """Make Python source file name."""
    target_file_name = os.path.splitext(os.path.basename(file_name))[0]
    target_file_name += '.' + ext
    target_file_name = os.path.join(dst_path, target_file_name)
    return target_file_name


def check_py_syntax(src_path, dst_path):
    """Check the Python syntax."""
    if not os.path.exists(src_path):
        os.makedirs(dst_path)
    good = []
    bad = []
    for file_name in glob.glob(os.path.join(src_path, '*.mochi')):
        mod_name = os.path.splitext(os.path.basename(file_name))[0]
        py_file_name = make_target_file_name(file_name, dst_path, 'py')
        with TempDir(src_path):
            try:
                make_py_source_file(mochi_file_name=file_name,
                                    python_file_name=py_file_name,
                                    mochi_env=MONKEY_PATCH_ENV,
                                    add_init=True, show_tokens=False)
            except TypeError as err:
                print('#' * 30)
                print('Error in module', mod_name)
                print('#' * 30)
                print(err)
                bad.append(mod_name)
                continue
        with TempDir(dst_path):
            try:
                py_compile.compile('{}.py'.format(mod_name), doraise=True)
            except Exception as err:
                print('#' * 30)
                print('Error in module', mod_name)
                print('#' * 30)
                print(err)
                bad.append(mod_name)
                continue
        good.append(mod_name)

    print('good', good)
    print('bad', bad)

if __name__ == '__main__':

    def check_examples():
        """Check Mochi files in example dir."""
        cwd = os.getcwd()
        src_path = os.path.normpath(os.path.join(cwd, '../examples'))
        dst_path = os.path.normpath(os.path.join(cwd, '../tmp'))
        check_py_syntax(src_path, dst_path)

    check_examples()
