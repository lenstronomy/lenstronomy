__author__ = "nataliehogg"

from lenstronomy.LightModel.Profiles.sersic_ellipse_with_flexion import (
    SersicEllipseWithFlexion,
)
import lenstronomy.Util.param_util as param_util
import numpy as np
import pytest
import numpy.testing as npt


class TestSersicEllipticWithFlexion(object):
    """Tests the elliptic flexed Sersic in the same way as the other Sersic profiles."""

    def setup_method(self):
        self.flexed_sersic = SersicEllipseWithFlexion(
            smoothing=0.02, sersic_major_axis=True
        )

    def test_sersic_elliptic_flexed(self):
        x = np.array([1])
        y = np.array([2])
        I0_sersic = 1
        R_sersic = 1
        n_sersic = 1
        phi_G = 1
        q = 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        center_x = 0
        center_y = 0
        values = self.flexed_sersic.function(
            x, y, I0_sersic, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.12595366113005077, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.flexed_sersic.function(
            x, y, I0_sersic, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 5.1482553482055664, decimal=2)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.flexed_sersic.function(
            x, y, I0_sersic, R_sersic, n_sersic, e1, e2, center_x, center_y
        )
        npt.assert_almost_equal(values[0], 0.11308277793465012, decimal=6)
        npt.assert_almost_equal(values[1], 0.021188620675507107, decimal=6)
        npt.assert_almost_equal(values[2], 0.0037276744362724477, decimal=6)


if __name__ == "__main__":
    pytest.main()
