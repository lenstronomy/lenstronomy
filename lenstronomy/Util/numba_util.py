import numba
import lenstronomy
import os
import yaml


"""
From pyautolens:
Depending on if we're using a super computer, we want two different numba decorators:
If on laptop:
@numba.jit(nopython=True, cache=True, parallel=False)
If on super computer:
@numba.jit(nopython=True, cache=False, parallel=True)
"""


# TODO define a special path outside the lenstronomy package to save a user configuration yaml file
user_config_file = ''
module_path = os.path.dirname(lenstronomy.__file__)
default_config_file = os.path.join(module_path, 'Conf', 'numba_conf_default.yaml')

if os.path.exists(user_config_file ):
    conf_file = user_config_file
else:
    conf_file = default_config_file

with open(conf_file) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    numba_list = yaml.load(file, Loader=yaml.FullLoader)
    nopython = numba_list['nopython']
    cache = numba_list['cache']
    parallel = numba_list['parallel']

#nopython = True
#cache = True
#parallel = False

__all__ = ['jit']


def jit(nopython=nopython, cache=cache, parallel=parallel):
    def wrapper(func):
        return numba.jit(func, nopython=nopython, cache=cache, parallel=parallel)

    return wrapper
