import numpy as np
import numpy.testing as npt
from lenstronomy.Sampling.Likelihoods.kinematic_2D_likelihood import KinLikelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.kinematic_bin_2D import KinBin
from lenstronomy.Data.pixel_grid import PixelGrid
import lenstronomy.Util.kernel_util as kernel_util
from lenstronomy.Sampling.Likelihoods import kinematic_NN_call


class TestKinLikelihood(object):
    def setup(self):
        self.kinematic_NN = kinematic_NN_call.KinematicNN().SKiNN_installed
        # initialize KinLikelihood class and calculate example vrms for testing
        major_axis_pa = 30 * np.pi / 180
        mass_q = 0.7
        light_q = 0.6
        center_x = 0.0
        center_y = 0.0
        self.kwargs_lens = [
            {
                "theta_E": 1.5,
                "gamma": 2,
                "q": mass_q,
                "phi": major_axis_pa,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        self.kwargs_lens_light = [
            {
                "amp": 10,
                "R_sersic": 1.0,
                "q": light_q,
                "phi": major_axis_pa,
                "n_sersic": 3.0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        self.kwargs_special = {
            "D_dt": 4000,
            "b_ani": 0.2,
            "incli": np.pi / 2,
            "D_d": 1000,
        }

        lens_model_list = ["EPL_Q_PHI"]
        self.lensModel = LensModel(lens_model_list=lens_model_list, z_lens=0.5)
        lens_light_model_list = ["SERSIC_ELLIPSE_Q_PHI"]

        self.lensLightModel = LightModel(lens_light_model_list)
        npix = 3
        delta_pix_image = 0.2
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix_image
        ra_at_xy_0 = -(npix - 1) / 2.0 * delta_pix_image
        dec_at_xy_0 = -(npix - 1) / 2.0 * delta_pix_image

        image_pixel_grid = PixelGrid(
            npix, npix, transform_pix2angle, ra_at_xy_0, dec_at_xy_0
        )
        self.image_data = self.lensLightModel.surface_brightness(
            image_pixel_grid._x_grid, image_pixel_grid._y_grid, self.kwargs_lens_light
        )

        binmap = np.array([[0, 0, 1], [0, 0, 1], [2, 2, 3]])
        binned_dummy_data = np.ones(4)
        delta_pix_kin = delta_pix_image
        npix_kin = npix

        # create PSF for kinematics
        sharp_kin_psf = kernel_util.kernel_gaussian(
            num_pix=npix, delta_pix=delta_pix_image, fwhm=0.001
        )

        # KinLikelihood inputs: light image kwargs, kinematic image kwargs, PSF, lens model class, light model class
        self.kwargs_data = {
            "image_data": self.image_data,
            "noise_map": 100 * np.ones((npix, npix)),
            "transform_pix2angle": transform_pix2angle,
            "ra_at_xy_0": ra_at_xy_0,
            "dec_at_xy_0": dec_at_xy_0,
        }
        self.kwargs_kin = {
            "bin_data": binned_dummy_data,
            "bin_cov": np.diag((binned_dummy_data * 0.05) ** 2),  # 5% error
            "bin_mask": binmap,
            "ra_at_xy_0": -(npix_kin - 1) / 2.0 * delta_pix_kin,
            "dec_at_xy_0": -(npix_kin - 1) / 2.0 * delta_pix_kin,
            "transform_pix2angle": np.array([[1, 0], [0, 1]]) * delta_pix_kin,
        }

        kwargs_pixelkin = {"psf_type": "PIXEL", "kernel_point_source": sharp_kin_psf}
        self.kinPSF = PSF(**kwargs_pixelkin)
        _KinBin = KinBin(psf_class=self.kinPSF, **self.kwargs_kin)
        self._KinLikelihood = KinLikelihood(
            _KinBin,
            self.lensModel,
            self.lensLightModel,
            self.kwargs_data,
            idx_lens=0,
            idx_lens_light=0,
        )
        if self.kinematic_NN:
            # if NN enabled, calculate realistic values to use as truth for logL tests
            self.truth_vrms = self._KinLikelihood.calc_vrms(
                self.kwargs_lens,
                self.kwargs_lens_light,
                self.kwargs_special,
                verbose=False,
            )
            self.kwargs_kin = {
                "bin_data": self.truth_vrms,
                "bin_cov": np.diag((self.truth_vrms * 0.05) ** 2),  # 5% error
                "bin_mask": binmap,
                "ra_at_xy_0": -(npix_kin - 1) / 2.0 * delta_pix_kin,
                "dec_at_xy_0": -(npix_kin - 1) / 2.0 * delta_pix_kin,
                "transform_pix2angle": np.array([[1, 0], [0, 1]]) * delta_pix_kin,
            }
            _KinBin = KinBin(psf_class=self.kinPSF, **self.kwargs_kin)
            self._KinLikelihood = KinLikelihood(
                _KinBin,
                self.lensModel,
                self.lensLightModel,
                self.kwargs_data,
                idx_lens=0,
                idx_lens_light=0,
            )

    def test_logL(self):
        if self.kinematic_NN:
            kwargs_lens_close = [{}]
            kwargs_lens_far = [{}]
            for key, val in self.kwargs_lens[0].items():
                kwargs_lens_close[0][key] = val + 0.01
                kwargs_lens_far[0][key] = val + 0.2
            kwargs_lens_light_close = [{}]
            kwargs_lens_light_far = [{}]
            for key, val in self.kwargs_lens_light[0].items():
                kwargs_lens_light_close[0][key] = val + 0.01
                kwargs_lens_light_far[0][key] = val + 0.2

            close_logL = self._KinLikelihood.logL(
                kwargs_lens_close,
                kwargs_lens_light_close,
                self.kwargs_special,
                verbose=False,
            )
            far_logL = self._KinLikelihood.logL(
                kwargs_lens_far,
                kwargs_lens_light_far,
                self.kwargs_special,
                verbose=False,
            )
            assert (
                close_logL > -10
            )  # instance close to truth should be likely (usually around -0.5)
            assert (
                close_logL > far_logL
            )  # param instance close to truth is more likely than far away

            # test  that doubling variance decreases logL by factor of 2
            kwargs_kin = self.kwargs_kin.copy()
            kwargs_kin["bin_cov"] = self.kwargs_kin["bin_cov"] * 2
            _KinBin = KinBin(psf_class=self.kinPSF, **kwargs_kin)
            var_kin_likelihood = KinLikelihood(
                _KinBin,
                self.lensModel,
                self.lensLightModel,
                self.kwargs_data,
                idx_lens=0,
                idx_lens_light=0,
            )
            var_logL = var_kin_likelihood.logL(
                kwargs_lens_close,
                kwargs_lens_light_close,
                self.kwargs_special,
                verbose=False,
            )
            npt.assert_almost_equal(var_logL, close_logL / 2, decimal=2)

            # test that if velocity off by 1 sigma, logL off by 1 sigma
            kwargs_kin = self.kwargs_kin.copy()
            kwargs_kin["bin_data"][-1] = (
                kwargs_kin["bin_data"][-1] * 1.05
            )  # increase one bin by 1 sigma

            _KinBin = KinBin(psf_class=self.kinPSF, **kwargs_kin)
            sig_kin_likelihood = KinLikelihood(
                _KinBin,
                self.lensModel,
                self.lensLightModel,
                self.kwargs_data,
                idx_lens=0,
                idx_lens_light=0,
            )
            sig_logL = sig_kin_likelihood.logL(
                kwargs_lens_close,
                kwargs_lens_light_close,
                self.kwargs_special,
                verbose=False,
            )
            npt.assert_almost_equal(sig_logL, close_logL - 1 / 2, decimal=1)
            kwargs_out_of_bounds = kwargs_lens_close.copy()
            kwargs_out_of_bounds[0]["theta_E"] = 10
            out_of_bounds_logL = self._KinLikelihood.logL(
                kwargs_lens_close,
                kwargs_lens_light_close,
                self.kwargs_special,
                verbose=False,
            )
            assert out_of_bounds_logL == -(10**8)

    def test_convert_to_nn_params(self):
        kwargs_lens_test = [
            {
                "theta_E": 2.0,
                "gamma": 2.0,
                "q": 1.0,
                "phi": 0,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.06, "gamma2": -0.03},
        ]
        kwargs_lens_light_test = [
            {
                "amp": 10,
                "R_sersic": 1.0,
                "q": 1,
                "phi": 0,
                "n_sersic": 3.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]
        params, same_orientation = self._KinLikelihood.convert_to_nn_params(
            kwargs_lens_test, kwargs_lens_light_test, self.kwargs_special
        )
        npt.assert_array_equal(
            params,
            np.array(
                [
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
                    8.0e-2,
                    0.5,
                    self.kwargs_special["b_ani"],
                    self.kwargs_special["incli"] * 180 / np.pi,
                ]
            ),
        )
        # check that wrong orientations are caught
        assert same_orientation == True
        kwargs_lens_test = [
            {
                "theta_E": 2.0,
                "gamma": 2.0,
                "q": 0.5,
                "phi": 1.0,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.06, "gamma2": -0.03},
        ]
        params, same_orientation = self._KinLikelihood.convert_to_nn_params(
            kwargs_lens_test, kwargs_lens_light_test, self.kwargs_special
        )
        assert same_orientation == False

    def test_rescale_distance(self):
        # scale=fiducial
        kwargs_special = {"D_dt": 2886.544, "b_ani": 0.1, "incli": 0.0, "D_d": 1215.739}
        rescaled_map = self._KinLikelihood.rescale_distance(
            self.image_data, kwargs_special
        )
        npt.assert_allclose(rescaled_map, self.image_data, atol=10**-4)
        # scale D_dt
        kwargs_special = {
            "D_dt": 2886.544 * 2,
            "b_ani": 0.1,
            "incli": 0.0,
            "D_d": 1215.739,
        }
        rescaled_map = self._KinLikelihood.rescale_distance(
            self.image_data, kwargs_special
        )
        npt.assert_allclose(np.sqrt(2) * self.image_data, rescaled_map, atol=10**-4)
        # scale D_d
        kwargs_special = {
            "D_dt": 2886.544,
            "b_ani": 0.1,
            "incli": 0.0,
            "D_d": 1215.739 * 2,
        }
        rescaled_map = self._KinLikelihood.rescale_distance(
            self.image_data, kwargs_special
        )
        npt.assert_allclose(1 / np.sqrt(2) * self.image_data, rescaled_map, atol=10**-4)

    def test_convert_kwargs_to_kinnalign_input(self):
        self._KinLikelihood.update_image_input(
            self.kwargs_lens
        )  # adds PA and centers to kwargs
        assert "ellipse_PA" in self._KinLikelihood.image_input

    def test_auto_binning(self):
        # light weighting testing, change light image
        dummy_vrms_map = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sharp_image = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) + 0.0001

        # auto_binning inherits the psf, bin map, and data shape
        sharp_kin_psf = kernel_util.kernel_gaussian(num_pix=5, delta_pix=0.2, fwhm=0.01)
        kwargs_pixelkin = {"psf_type": "PIXEL", "kernel_point_source": sharp_kin_psf}
        kin_psf = PSF(**kwargs_pixelkin)
        _KinBin = KinBin(psf_class=kin_psf, **self.kwargs_kin)
        _KinLikelihood = KinLikelihood(
            _KinBin,
            self.lensModel,
            self.lensLightModel,
            self.kwargs_data,
            idx_lens=0,
            idx_lens_light=0,
        )
        vrms = _KinLikelihood.auto_binning(dummy_vrms_map, sharp_image)
        npt.assert_allclose(vrms, [5, 4.5, 7.5, 9], rtol=1e-3)
        vrms = _KinLikelihood.auto_binning(dummy_vrms_map, np.ones((3, 3)))
        npt.assert_allclose(vrms, [3, 4.5, 7.5, 9], rtol=1e-3)

        wide_kin_psf = kernel_util.kernel_gaussian(num_pix=5, delta_pix=0.2, fwhm=100)
        kwargs_pixelkin = {"psf_type": "PIXEL", "kernel_point_source": wide_kin_psf}
        kin_psf = PSF(**kwargs_pixelkin)
        _KinBin = KinBin(psf_class=kin_psf, **self.kwargs_kin)
        _KinLikelihood = KinLikelihood(
            _KinBin,
            self.lensModel,
            self.lensLightModel,
            self.kwargs_data,
            idx_lens=0,
            idx_lens_light=0,
        )

        vrms = _KinLikelihood.auto_binning(dummy_vrms_map, sharp_image)
        expected = np.mean(dummy_vrms_map) / np.mean(sharp_image) * np.ones(4) / 25.0
        npt.assert_allclose(vrms, expected, rtol=1e-3)
