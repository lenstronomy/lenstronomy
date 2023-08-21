import pytest
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics


class TestAnalyticKinematics(object):
    def setup_method(self):
        pass

    def test_sigma_s2(self):
        kwargs_aperture = {
            "center_ra": 0,
            "width": 1,
            "length": 1,
            "angle": 0,
            "center_dec": 0,
            "aperture_type": "slit",
        }
        kwargs_cosmo = {"d_d": 1000, "d_s": 1500, "d_ds": 800}
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": 1}
        kin = AnalyticKinematics(kwargs_cosmo)
        kwargs_light = {"r_eff": 1}
        sigma_s2 = kin.sigma_s2(
            r=1,
            R=0.1,
            kwargs_mass={"theta_E": 1, "gamma": 2},
            kwargs_light=kwargs_light,
            kwargs_anisotropy={"r_ani": 1},
        )
        assert "a" in kwargs_light


if __name__ == "__main__":
    pytest.main()
