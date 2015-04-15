import pytest


@pytest.fixture(scope="session")
def auto_test(mod):
    result_names = [name for name in dir(mod) if name.startswith('result_')]
    func_names = [name.split('_', 1)[1] for name in result_names]
    results = (getattr(mod, name) for name in result_names)
    funcs = (getattr(mod, name) for name in func_names)
    return zip(funcs, results)