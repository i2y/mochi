import sys
from pathlib import Path
from .pycloader import get_module


def is_package(full_name):
    module_path = '/'.join(full_name.split('.'))
    for load_path in sys.path:
        load_path = Path(load_path).absolute()
        target_path = Path('%s/%s/__init__.mochi'
                           % (load_path, module_path))
        if target_path.exists():
            return True
    return False


class Loader(object):
    def __init__(self, path):
        self.path = path

    def load_module(self, full_name):
        if full_name in sys.modules:
            return sys.modules[full_name]

        absolute_file_path = str(self.path.absolute())
        mod = get_module(full_name, absolute_file_path)
        print(">>>>>>>>>>>")
        print(absolute_file_path)
        print("<<<<<<<<<<<")
        mod.__file__ = absolute_file_path
        mod.__loader__ = self
        mod.__name__ = full_name

        if is_package(full_name):
            mod.__path__ = []
            mod.__package__ = full_name
        else:
            mod.__package__ = full_name.rpartition('.')[0]

        sys.modules[full_name] = mod
        return mod


def find_file(full_name):
    target_files = ['%s/__init__.mochi',
                    '%s.mochi']
    module_path = '/'.join(full_name.split('.'))

    for load_path in sys.path:
        load_path = Path(load_path).absolute()
        for target_file in target_files:
            target_path = Path(target_file % ('%s/%s'
                                              % (load_path, module_path)))
            if target_path.exists():
                return target_path


class Importer(object):
    def find_module(self, fullname, path=None):
        file_path = find_file(fullname)
        if file_path:
            pyc_file_path = Path(file_path.with_suffix('.pyc'))
            if pyc_file_path.exists():
                pyc_file_mtime = pyc_file_path.stat().st_mtime
                mochi_file_mtime = file_path.stat().st_mtime
                if mochi_file_mtime > pyc_file_mtime:
                    return Loader(file_path)
            else:
                return Loader(file_path)


def set_importer():
    sys.meta_path.insert(0, Importer())
    sys.path.insert(0, '')
