from importlib.util import find_spec


def is_module_exist(module_name):
    spec = find_spec(module_name)
    if spec is None:
        return False
    else:
        return True
