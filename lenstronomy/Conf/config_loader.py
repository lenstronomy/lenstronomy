import yaml
import os
import lenstronomy


# in case the xdg library is installed, the import statement with pyxdg can raise an error
# to avoid it, we draw back to the ~/.config directory in case this import fails.
# TODO come up with more permanent solution of path to configuration directory
try:
    from xdg.BaseDirectory import xdg_config_home
except ImportError:
    xdg_config_home = '~/.config'

user_config_file = os.path.join(xdg_config_home, "lenstronomy", "config.yaml")

module_path = os.path.dirname(lenstronomy.__file__)
default_config_file = os.path.join(module_path, 'Conf', 'conf_default.yaml')

if os.path.exists(user_config_file ):
    conf_file = user_config_file
else:
    conf_file = default_config_file

with open(conf_file) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to the Python the dictionary format
    conf = yaml.safe_load(file)
    #conf = yaml.load(file, Loader=yaml.FullLoader)
    numba_conf = conf['numba']
    nopython = numba_conf['nopython']
    cache = numba_conf['cache']
    parallel = numba_conf['parallel']
    numba_enabled = numba_conf['enable']
    fastmath = numba_conf['fastmath']
    error_model = numba_conf['error_model']


def numba_conf():
    """

    :return: keyword arguments of numba configurations from yaml file
    """
    with open(conf_file) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to the Python the dictionary format
        conf = yaml.safe_load(file)
        # conf = yaml.load(file, Loader=yaml.FullLoader)
        numba_conf = conf['numba']
    return numba_conf


def conventions_conf():
    """

    :return: convention keyword arguments
    """
    with open(conf_file) as file:
        conf = yaml.safe_load(file)
        conventions_conf = conf['conventions']
    return conventions_conf
