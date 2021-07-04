
import pytest
from lenstronomy.Conf import config_loader


def test_numba_conf():
    numba_conf = config_loader.numba_conf()
    assert 'nopython' in numba_conf
    assert 'cache' in numba_conf
    assert 'parallel' in numba_conf
    assert 'enable' in numba_conf
    assert 'fastmath' in numba_conf
    assert 'error_model' in numba_conf


def test_conventions_conf():
    conf = config_loader.conventions_conf()
    assert 'sersic_major_axis' in conf


if __name__ == '__main__':
    pytest.main()
