from lenstronomy.GalKin.galkin import Galkin
import numpy.testing as npt
import numpy as np


class TestGOM(object):
    def setup_method(self):
        np.random.seed(2)

    def test_OMvsGOM(self):
        """Test OsivkopMerrit vs generalized OM model :return:"""
        light_profile_list = ["HERNQUIST"]
        r_eff = 1.5
        kwargs_light = [
            {"Rs": r_eff, "amp": 1.0}
        ]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ["SPP"]
        theta_E = 1.2
        gamma = 2.0
        kwargs_profile = [
            {"theta_E": theta_E, "gamma": gamma}
        ]  # Einstein radius (arcsec) and power-law slope

        # aperture as slit
        aperture_type = "slit"
        length = 1.0
        width = 0.3
        kwargs_aperture = {
            "aperture_type": aperture_type,
            "length": length,
            "width": width,
            "center_ra": 0,
            "center_dec": 0,
            "angle": 0,
        }

        psf_fwhm = 1.0  # Gaussian FWHM psf
        kwargs_cosmo = {"d_d": 1000, "d_s": 1500, "d_ds": 800}
        kwargs_numerics = {
            "interpol_grid_num": 2000,
            "log_integration": True,
            "max_integrate": 100,
            "min_integrate": 0.00001,
        }

        # anisotropy profile
        anisotropy_type = "OM"
        r_ani = 0.2
        kwargs_anisotropy = {"r_ani": r_ani}  # anisotropy radius [arcsec]

        kwargs_model = {
            "mass_profile_list": mass_profile_list,
            "light_profile_list": light_profile_list,
            "anisotropy_model": anisotropy_type,
        }
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        galkin = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics,
        )
        sigma_v_om = galkin.dispersion(
            kwargs_profile, kwargs_light, kwargs_anisotropy, sampling_number=5000
        )

        # anisotropy profile
        anisotropy_type = "GOM"

        kwargs_anisotropy = {
            "r_ani": r_ani,
            "beta_inf": 1,
        }  # anisotropy radius [arcsec]

        kwargs_model = {
            "mass_profile_list": mass_profile_list,
            "light_profile_list": light_profile_list,
            "anisotropy_model": anisotropy_type,
        }
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": psf_fwhm}
        galkin_gom = Galkin(
            kwargs_model=kwargs_model,
            kwargs_aperture=kwargs_aperture,
            kwargs_psf=kwargs_psf,
            kwargs_cosmo=kwargs_cosmo,
            kwargs_numerics=kwargs_numerics,
        )
        sigma_v_gom = galkin_gom.dispersion(
            kwargs_profile, kwargs_light, kwargs_anisotropy, sampling_number=5000
        )
        # warning: this tests does not work to this precision for every random seed. To increase precision, increase
        # sampling_number
        npt.assert_almost_equal(sigma_v_gom / sigma_v_om, 1, decimal=2)
