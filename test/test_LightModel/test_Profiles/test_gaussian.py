
import pytest
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian


class TestMoffat(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_function_split(self):
        """

        :return:
        """
        profile = MultiGaussian()
        output = profile.function_split(x=1., y=1., amp=[1., 2], sigma=[1, 2], center_x=0, center_y=0)
        npt.assert_almost_equal(output[0], 0.058549831524319168, decimal=8)
        npt.assert_almost_equal(output[1], 0.061974997154826489, decimal=8)


if __name__ == '__main__':
    pytest.main()
