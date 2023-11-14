import pytest
from lenstronomy.GalKin.galkin_multiobservation import GalkinMultiObservation


class TestGalkinMultiObservation(object):
    def setup_method(self):
        pass

    def test_dispersion(self):
        light_profile_list = ["HERNQUIST"]
        Rs = 0.5
        kwargs_light = [
            {"Rs": Rs, "amp": 1.0}
        ]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ["SPP"]
        theta_E = 1.2
        gamma = 2.0
        kwargs_mass = [
            {"theta_E": theta_E, "gamma": gamma}
        ]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = "OM"
        r_ani = 2.0
        kwargs_anisotropy = {"r_ani": r_ani}  # anisotropy radius [arcsec]

        kwargs_model = {
            "mass_profile_list": mass_profile_list,
            "light_profile_list": light_profile_list,
            "anisotropy_model": anisotropy_type,
        }
        kwargs_cosmo = {"d_d": 1000, "d_s": 1500, "d_ds": 800}
        kwargs_numerics = {
            "interpol_grid_num": 500,
            "log_integration": True,
            "max_integrate": 10,
            "min_integrate": 0.001,
        }

        # aperture as slit
        aperture_type = "slit"
        kwargs_aperture_1 = {"width": 1, "length": 1.0, "aperture_type": aperture_type}
        kwargs_psf_1 = {"psf_type": "GAUSSIAN", "fwhm": 0.7}

        kwargs_aperture_2 = {"width": 3, "length": 3.0, "aperture_type": aperture_type}
        kwargs_psf_2 = {"psf_type": "GAUSSIAN", "fwhm": 1.5}
        kwargs_aperture_list = [kwargs_aperture_1, kwargs_aperture_2]
        kwargs_psf_list = [kwargs_psf_1, kwargs_psf_2]
        galkin_multiobs = GalkinMultiObservation(
            kwargs_model,
            kwargs_aperture_list,
            kwargs_psf_list,
            kwargs_cosmo,
            kwargs_numerics=kwargs_numerics,
            analytic_kinematics=False,
        )

        sigma_v_list = galkin_multiobs.dispersion_map(
            kwargs_mass=kwargs_mass,
            kwargs_light=kwargs_light,
            kwargs_anisotropy=kwargs_anisotropy,
            num_kin_sampling=1000,
            num_psf_sampling=100,
        )
        assert len(sigma_v_list) == 2
        assert sigma_v_list[0] > sigma_v_list[1]


if __name__ == "__main__":
    pytest.main()
