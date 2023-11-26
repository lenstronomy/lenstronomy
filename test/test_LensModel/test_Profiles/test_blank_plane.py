import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LensModel.Profiles.blank_plane import BlankPlane


class TestBlankPlane(object):
    """Class to test the "BLANK_PLANE" lens model."""

    def setup_method(self):
        self.blank = BlankPlane()

    def test_function(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        f_ = self.blank.function(x, x)
        npt.assert_almost_equal(f_, x * 0, decimal=5)

    def test_derivatives(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        f_x_1, f_y_1 = self.blank.derivatives(x, x)
        npt.assert_almost_equal(f_x_1, x * 0, decimal=5)
        npt.assert_almost_equal(f_y_1, x * 0, decimal=5)

    def test_hessian(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        f_xx, f_xy, f_yx, f_yy = self.blank.hessian(x, x)
        npt.assert_almost_equal(f_xx, x * 0, decimal=5)
        npt.assert_almost_equal(f_xy, x * 0, decimal=5)
        npt.assert_almost_equal(f_yx, x * 0, decimal=5)
        npt.assert_almost_equal(f_yy, x * 0, decimal=5)


if __name__ == "__main__":
    pytest.main()
