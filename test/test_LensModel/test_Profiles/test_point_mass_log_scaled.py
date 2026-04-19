__author__ = "ajshajib"


from lenstronomy.LensModel.Profiles.point_mass import PointMass
from lenstronomy.LensModel.Profiles.point_mass_log_scaled import PointMassLogScaled
from lenstronomy.LensModel.profile_list_base import lens_class

import numpy as np
import numpy.testing as npt
import pytest


class TestPointMassLogScaled(object):
    """Tests the PointMassLogScaled class routines."""

    def setup_method(self):
        self.pointmass = PointMass()
        self.pointmass_log_scaled = PointMassLogScaled()
        self.theta_E = 1.7
        self.log10_theta_E = np.log10(self.theta_E)

    def test_function(self):
        x = np.array([1.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 1.0])
        values = self.pointmass_log_scaled.function(x, y, self.log10_theta_E)
        values_ref = self.pointmass.function(x, y, self.theta_E)
        npt.assert_allclose(values, values_ref)

    def test_derivatives(self):
        x = np.array([1.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 1.0])
        values = self.pointmass_log_scaled.derivatives(x, y, self.log10_theta_E)
        values_ref = self.pointmass.derivatives(x, y, self.theta_E)
        for value, value_ref in zip(values, values_ref):
            npt.assert_allclose(value, value_ref)

    def test_hessian(self):
        x = np.array([1.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 1.0])
        values = self.pointmass_log_scaled.hessian(x, y, self.log10_theta_E)
        values_ref = self.pointmass.hessian(x, y, self.theta_E)
        for value, value_ref in zip(values, values_ref):
            npt.assert_allclose(value, value_ref)

    def test_mass_3d_lens(self):
        mass_3d = self.pointmass_log_scaled.mass_3d_lens(5.0, self.log10_theta_E)
        mass_3d_ref = self.pointmass.mass_3d_lens(5.0, self.theta_E)
        assert mass_3d == mass_3d_ref

    def test_registered_lens_class(self):
        point_mass_log_scaled = lens_class("POINT_MASS_LOG_SCALED")
        assert isinstance(point_mass_log_scaled, PointMassLogScaled)
        assert point_mass_log_scaled.param_names == [
            "log10_theta_E",
            "center_x",
            "center_y",
        ]


if __name__ == "__main__":
    pytest.main()
