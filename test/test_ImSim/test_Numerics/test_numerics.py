import pytest
import numpy.testing as npt
import numpy as np
import unittest

import lenstronomy.Util.util as util
import lenstronomy.Util.kernel_util as kernel_util
from lenstronomy.ImSim.image_model import ImageModel


class TestNumerics(object):
    def setup_method(self):
        # we define a model consisting of a singe Sersric profile
        from lenstronomy.LightModel.light_model import LightModel

        light_model_list = ["SERSIC_ELLIPSE"]
        self.lightModel = LightModel(light_model_list=light_model_list)
        self.kwargs_light = [
            {
                "amp": 100,
                "R_sersic": 0.5,
                "n_sersic": 3,
                "e1": 0,
                "e2": 0,
                "center_x": 0.02,
                "center_y": 0,
            }
        ]

        # we define a pixel grid and a higher resolution super sampling factor
        self._supersampling_factor = 5
        numPix = 61  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        (
            x,
            y,
            ra_at_xy_0,
            dec_at_xy_0,
            x_at_radec_0,
            y_at_radec_0,
            Mpix2coord,
            Mcoord2pix,
        ) = util.make_grid_with_coordtransform(
            numPix=numPix,
            deltapix=deltaPix,
            subgrid_res=1,
            left_lower=False,
            inverse=False,
        )
        flux = self.lightModel.surface_brightness(x, y, kwargs_list=self.kwargs_light)
        flux = util.array2image(flux)
        flux_max = np.max(flux)
        conv_pixels_partial = np.zeros((numPix, numPix), dtype=bool)
        conv_pixels_partial[flux >= flux_max / 20] = True
        self._conv_pixels_partial = conv_pixels_partial

        # high resolution ray-tracing and high resolution convolution, the full calculation
        self.kwargs_numerics_true = {
            "supersampling_factor": self._supersampling_factor,
            # super sampling factor of (partial) high resolution ray-tracing
            "compute_mode": "regular",  # 'regular' or 'adaptive'
            "supersampling_convolution": True,
            # bool, if True, performs the supersampled convolution (either on regular or adaptive grid)
            "supersampling_kernel_size": None,
            # size of the higher resolution kernel region (can be smaller than the original kernel). None leads to use the full size
            "flux_evaluate_indexes": None,  # bool mask, if None, it will evaluate all (sub) pixels
            "supersampled_indexes": None,
            # bool mask of pixels to be computed in supersampled grid (only for adaptive mode)
            "compute_indexes": None,
            # bool mask of pixels to be computed the PSF response (flux being added to). Only used for adaptive mode and can be set =likelihood mask.
            "point_source_supersampling_factor": 1,
            # int, supersampling factor when rendering a point source (not used in this script)
        }

        # high resolution convolution on a smaller PSF with low resolution convolution on the edges of the PSF and high resolution ray tracing
        self.kwargs_numerics_high_res_narrow = {
            "supersampling_factor": self._supersampling_factor,
            "compute_mode": "regular",
            "supersampling_convolution": True,
            "supersampling_kernel_size": 5,
        }

        # low resolution convolution based on high resolution ray-tracing grid
        self.kwargs_numerics_low_conv_high_grid = {
            "supersampling_factor": self._supersampling_factor,
            "compute_mode": "regular",
            "supersampling_convolution": False,
            # does not matter for supersampling_factor=1
            "supersampling_kernel_size": None,
            # does not matter for supersampling_factor=1
        }

        # low resolution convolution with a subset of pixels with high resolution ray-tracing
        self.kwargs_numerics_low_conv_high_adaptive = {
            "supersampling_factor": self._supersampling_factor,
            "compute_mode": "adaptive",
            "supersampling_convolution": False,
            # does not matter for supersampling_factor=1
            "supersampling_kernel_size": None,
            # does not matter for supersampling_factor=1
            "supersampled_indexes": self._conv_pixels_partial,
            "convolution_kernel_size": 9,
        }

        # low resolution convolution with a subset of pixels with high resolution ray-tracing and high resoluton convolution on smaller kernel size
        self.kwargs_numerics_high_adaptive = {
            "supersampling_factor": self._supersampling_factor,
            "compute_mode": "adaptive",
            "supersampling_convolution": True,
            # does not matter for supersampling_factor=1
            "supersampling_kernel_size": 5,  # does not matter for supersampling_factor=1
            "supersampled_indexes": self._conv_pixels_partial,
            "convolution_kernel_size": 9,
        }

        # low resolution convolution and low resolution ray tracing, the simplest calculation
        self.kwargs_numerics_low_res = {
            "supersampling_factor": 1,
            "compute_mode": "regular",
            "supersampling_convolution": False,  # does not matter for supersampling_factor=1
            "supersampling_kernel_size": None,  # does not matter for supersampling_factor=1
            "convolution_kernel_size": 9,
        }

        flux_evaluate_indexes = np.zeros((numPix, numPix), dtype=bool)
        flux_evaluate_indexes[flux >= flux_max / 1000] = True
        # low resolution convolution on subframe
        self.kwargs_numerics_partial = {
            "supersampling_factor": 1,
            "compute_mode": "regular",
            "supersampling_convolution": False,
            # does not matter for supersampling_factor=1
            "supersampling_kernel_size": None,  # does not matter for supersampling_factor=1
            "flux_evaluate_indexes": flux_evaluate_indexes,
            "convolution_kernel_size": 9,
        }

        # import PSF file
        kernel_super = kernel_util.kernel_gaussian(
            num_pix=11 * self._supersampling_factor,
            delta_pix=deltaPix / self._supersampling_factor,
            fwhm=0.1,
        )

        kernel_size = 9
        kernel_super = kernel_util.cut_psf(
            psf_data=kernel_super, psf_size=kernel_size * self._supersampling_factor
        )

        # make instance of the PixelGrid class
        from lenstronomy.Data.pixel_grid import PixelGrid

        kwargs_grid = {
            "nx": numPix,
            "ny": numPix,
            "transform_pix2angle": Mpix2coord,
            "ra_at_xy_0": ra_at_xy_0,
            "dec_at_xy_0": dec_at_xy_0,
        }
        self.pixel_grid = PixelGrid(**kwargs_grid)

        # make instance of the PSF class
        from lenstronomy.Data.psf import PSF

        kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": kernel_super,
            "point_source_supersampling_factor": self._supersampling_factor,
        }
        self.psf_class = PSF(**kwargs_psf)

        # without convolution
        image_model_true = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_true,
        )
        self.image_true = image_model_true.image(kwargs_lens_light=self.kwargs_light)

    def test_full(self):
        image_model_true = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_true,
        )
        image_unconvolved = image_model_true.image(
            kwargs_lens_light=self.kwargs_light, unconvolved=True
        )
        npt.assert_almost_equal(
            np.sum(self.image_true) / np.sum(image_unconvolved), 1, decimal=2
        )

    def test_high_res_narrow(self):
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_high_res_narrow,
        )
        image_conv = image_model.image(
            kwargs_lens_light=self.kwargs_light, unconvolved=False
        )
        npt.assert_almost_equal(
            (self.image_true - image_conv) / self.image_true, 0, decimal=2
        )

    def test_low_conv_high_grid(self):
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_low_conv_high_grid,
        )
        image_conv = image_model.image(
            kwargs_lens_light=self.kwargs_light, unconvolved=False
        )
        npt.assert_almost_equal(
            (self.image_true - image_conv) / self.image_true, 0, decimal=1
        )

    def test_low_conv_high_adaptive(self):
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_low_conv_high_adaptive,
        )
        image_conv = image_model.image(
            kwargs_lens_light=self.kwargs_light, unconvolved=False
        )
        npt.assert_almost_equal(
            (self.image_true - image_conv) / self.image_true, 0, decimal=1
        )

    def test_high_adaptive(self):
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_high_adaptive,
        )
        image_conv = image_model.image(
            kwargs_lens_light=self.kwargs_light, unconvolved=False
        )
        npt.assert_almost_equal(
            (self.image_true - image_conv) / self.image_true, 0, decimal=1
        )

    def test_low_res(self):
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_low_res,
        )
        image_conv = image_model.image(
            kwargs_lens_light=self.kwargs_light, unconvolved=False
        )
        npt.assert_almost_equal(
            (self.image_true - image_conv) / self.image_true, 0, decimal=1
        )

    def test_sub_frame(self):
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_partial,
        )
        image_conv = image_model.image(
            kwargs_lens_light=self.kwargs_light, unconvolved=False
        )
        delta = (self.image_true - image_conv) / self.image_true
        npt.assert_almost_equal(delta[self._conv_pixels_partial], 0, decimal=1)

    def test_property_access(self):
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=self.kwargs_numerics_true,
        )
        grid_supersampling_factor = image_model.ImageNumerics.grid_supersampling_factor
        assert grid_supersampling_factor == self._supersampling_factor

        kwargs_numerics = {
            "supersampling_factor": 1,
            "compute_mode": "regular",
            "supersampling_convolution": False,
        }
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=kwargs_numerics,
        )
        from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution

        convolution_class = image_model.ImageNumerics.convolution_class
        assert isinstance(convolution_class, PixelKernelConvolution)

        kwargs_numerics = {
            "supersampling_factor": 2,
            "compute_mode": "regular",
            "supersampling_convolution": True,
        }
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=kwargs_numerics,
        )
        from lenstronomy.ImSim.Numerics.convolution import SubgridKernelConvolution

        convolution_class = image_model.ImageNumerics.convolution_class
        assert isinstance(convolution_class, SubgridKernelConvolution)

        kwargs_numerics = {
            "supersampling_factor": 2,
            "compute_mode": "adaptive",
            "supersampling_convolution": True,
        }
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=kwargs_numerics,
        )
        from lenstronomy.ImSim.Numerics.adaptive_numerics import AdaptiveConvolution

        convolution_class = image_model.ImageNumerics.convolution_class
        assert isinstance(convolution_class, AdaptiveConvolution)

        kwargs_numerics = {"compute_mode": "regular"}
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=kwargs_numerics,
        )
        from lenstronomy.ImSim.Numerics.grid import RegularGrid

        grid_class = image_model.ImageNumerics.grid_class
        assert isinstance(grid_class, RegularGrid)

        kwargs_numerics = {"compute_mode": "adaptive"}
        image_model = ImageModel(
            self.pixel_grid,
            self.psf_class,
            lens_light_model_class=self.lightModel,
            kwargs_numerics=kwargs_numerics,
        )
        from lenstronomy.ImSim.Numerics.grid import AdaptiveGrid

        grid_class = image_model.ImageNumerics.grid_class
        assert isinstance(grid_class, AdaptiveGrid)


def test_supersampling_simple():
    """

    :return:
    """
    from lenstronomy.Data.psf import PSF
    from lenstronomy.SimulationAPI.data_api import DataAPI

    detector_pixel_scale = 0.04
    numpix = 64
    supersampling_factor = 2
    # generate a Gaussian image

    x, y = util.make_grid(
        numPix=numpix * supersampling_factor,
        deltapix=detector_pixel_scale / supersampling_factor,
    )
    from lenstronomy.LightModel.Profiles.gaussian import Gaussian

    gaussian = Gaussian()
    image_1d = gaussian.function(x, y, amp=1, sigma=0.1)
    image = util.array2image(image_1d)

    # generate psf kernal supersampled
    kernel_super = kernel_util.kernel_gaussian(
        num_pix=21 * supersampling_factor + 1,
        delta_pix=detector_pixel_scale / supersampling_factor,
        fwhm=0.2,
    )

    psf_parameters = {
        "psf_type": "PIXEL",
        "kernel_point_source": kernel_super,
        "point_source_supersampling_factor": supersampling_factor,
    }
    kwargs_detector = {
        "pixel_scale": detector_pixel_scale,
        "ccd_gain": 2.5,
        "read_noise": 4.0,
        "magnitude_zero_point": 25.0,
        "exposure_time": 5400.0,
        "sky_brightness": 22,
        "num_exposures": 1,
        "background_noise": None,
    }
    kwargs_numerics = {
        "supersampling_factor": 2,
        "supersampling_convolution": True,
        "point_source_supersampling_factor": 2,
        "supersampling_kernel_size": 21,
    }
    psf_model = PSF(**psf_parameters)
    data_class = DataAPI(numpix=numpix, **kwargs_detector).data_class

    from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame

    image_numerics = NumericsSubFrame(
        pixel_grid=data_class, psf=psf_model, **kwargs_numerics
    )

    conv_class = image_numerics.convolution_class
    conv_flat = conv_class.convolution2d(image)
    print(np.shape(conv_flat), "shape of output")

    # psf_helper = lenstronomy_utils.PSFHelper(data_class, psf_model, kwargs_numerics)

    # Convolve with lenstronomy and with scipy
    # helper_image = psf_helper.psf_model(image)
    from scipy import signal

    scipy_image = signal.fftconvolve(image, kernel_super, mode="same")
    from lenstronomy.Util import image_util

    image_scipy_resized = image_util.re_size(scipy_image, supersampling_factor)
    image_unconvolved = image_util.re_size(image, supersampling_factor)

    # Compare the outputs

    # low res convolution as comparison
    kwargs_numerics_low_res = {
        "supersampling_factor": 2,
        "supersampling_convolution": False,
        "point_source_supersampling_factor": 2,
    }
    image_numerics_low_res = NumericsSubFrame(
        pixel_grid=data_class, psf=psf_model, **kwargs_numerics_low_res
    )
    conv_class_low_res = image_numerics_low_res.convolution_class
    conv_flat_low_res = conv_class_low_res.convolution2d(image_unconvolved)

    # import matplotlib.pyplot as plt
    # plt.matshow(image_scipy_resized - image_unconvolved)
    # plt.colorbar()
    # plt.show()

    # plt.matshow(image_scipy_resized - conv_flat)
    # plt.colorbar()
    # plt.show()

    # plt.matshow(image_scipy_resized - conv_flat_low_res)
    # plt.colorbar()
    # plt.show()

    np.testing.assert_almost_equal(conv_flat, image_scipy_resized)


class TestRaise(unittest.TestCase):
    def test_integer_in_supersampling_factor(self):
        from lenstronomy.Data.psf import PSF

        kwargs_psf = {"psf_type": "NONE"}
        psf_class = PSF(**kwargs_psf)

        from lenstronomy.ImSim.Numerics.numerics import Numerics

        with self.assertRaises(TypeError):
            Numerics(pixel_grid=None, psf=psf_class, supersampling_factor=1.0)


if __name__ == "__main__":
    pytest.main()
