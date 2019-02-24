import numpy.testing as npt
import pytest
import lenstronomy.Util.data_util as data_util


def test_absolute2apparent_magnitude():
    absolute_magnitude = 0
    d_parsec = 10
    apparent_magnitude = data_util.absolute2apparent_magnitude(absolute_magnitude, d_parsec)
    npt.assert_almost_equal(apparent_magnitude, 0, decimal=8)


if __name__ == '__main__':
    pytest.main()
