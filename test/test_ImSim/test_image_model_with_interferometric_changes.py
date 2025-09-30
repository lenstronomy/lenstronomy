import numpy.testing as npt
import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import scipy.signal

"""
Test the implementation of interferometric PSF and primary beam in image simulation.
"""


def test_lens_light_and_source_light_simulation_with_interferometric_PSF_and_primary_beam():
    sigma_bkg = 0.05
    exp_time = np.inf
    numPix = 100
    deltaPix = 0.05

    # simulate a primary beam (pb)
    primary_beam = np.zeros((numPix, numPix))
    for i in range(numPix):
        for j in range(numPix):
            primary_beam[i, j] = np.exp(-1e-4 * ((i - 78) ** 2 + (j - 56) ** 2))
    primary_beam /= np.max(primary_beam)

    # simulate a spherical sinc function as psf, which contains negative pixels
    psf_test = np.zeros((221, 221))
    for i in range(221):
        for j in range(221):
            if i > j:
                psf_test[i, j] = psf_test[j, i]
            r = np.sqrt((i - 110) ** 2 + (j - 110) ** 2)
            if r == 0:
                psf_test[i, j] = 1
            else:
                psf_test[i, j] = np.sin(r * 0.5) / (r * 0.5)

    # define two data classes
    kwargs_data_no_pb = sim_util.data_configure_simple(
        numPix, deltaPix, exp_time, sigma_bkg
    )
    data_class_no_pb = ImageData(**kwargs_data_no_pb)

    kwargs_data_with_pb = sim_util.data_configure_simple(
        numPix, deltaPix, exp_time, sigma_bkg
    )
    kwargs_data_with_pb["antenna_primary_beam"] = primary_beam
    data_class_with_pb = ImageData(**kwargs_data_with_pb)

    # define two psf classes
    kwargs_psf_none = {"psf_type": "NONE"}
    psf_class_none = PSF(**kwargs_psf_none)

    kernel_cut = kernel_util.cut_psf(psf_test, 201, normalisation=False)
    kwargs_psf = {
        "psf_type": "PIXEL",
        "pixel_size": deltaPix,
        "kernel_point_source": kernel_cut,
        "kernel_point_source_normalisation": False,
    }
    psf_class = PSF(**kwargs_psf)

    # define lens model and source model
    kwargs_shear = {"gamma1": 0.01, "gamma2": 0.01}
    kwargs_spemd = {
        "theta_E": 1.0,
        "gamma": 1.8,
        "center_x": 0,
        "center_y": 0,
        "e1": 0.1,
        "e2": 0.04,
    }
    lens_model_list = ["SPEP", "SHEAR"]
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)

    kwargs_sersic = {
        "amp": 30.0,
        "R_sersic": 0.3,
        "n_sersic": 2,
        "center_x": 0,
        "center_y": 0,
    }
    lens_light_model_list = ["SERSIC"]
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

    kwargs_sersic_ellipse = {
        "amp": 1.0,
        "R_sersic": 0.6,
        "n_sersic": 7,
        "center_x": 0,
        "center_y": 0,
        "e1": 0.05,
        "e2": 0.02,
    }
    source_model_list = ["SERSIC_ELLIPSE"]
    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)

    kwargs_numerics = {"supersampling_factor": 1, "supersampling_convolution": False}

    # make images using 1) data and psf classes without pb and or psf
    imageModel_no_pb_psf = ImageModel(
        data_class_no_pb,
        psf_class_none,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        kwargs_numerics=kwargs_numerics,
    )
    image_sim_no_pb_psf = imageModel_no_pb_psf.image(
        kwargs_lens, kwargs_source, kwargs_lens_light
    )

    # make images using 2) data and psf classes with defined pb and or psf
    imageModel_with_pb_psf = ImageModel(
        data_class_with_pb,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        kwargs_numerics=kwargs_numerics,
    )
    image_sim_with_pb_psf = imageModel_with_pb_psf.image(
        kwargs_lens, kwargs_source, kwargs_lens_light
    )

    # add pb and psf to 1) out of the imageModel, compare them to check if the pb and psf changes make sense
    image_sim_with_pb_psf2 = scipy.signal.fftconvolve(
        image_sim_no_pb_psf * primary_beam, kernel_cut, mode="same"
    )
    npt.assert_almost_equal(image_sim_with_pb_psf, image_sim_with_pb_psf2, decimal=8)

    # test the "apply_primary_beam" parameter
    imageModel_with_psf_without_pb = ImageModel(
        data_class_no_pb,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        kwargs_numerics=kwargs_numerics,
    )

    ## test "apply_primary_beam" of source_surface_brightness function
    source_without_pb_image1 = imageModel_with_psf_without_pb.source_surface_brightness(
        kwargs_source, de_lensed=True, unconvolved=True
    )
    source_without_pb_image2 = imageModel_with_pb_psf.source_surface_brightness(
        kwargs_source, de_lensed=True, unconvolved=True, apply_primary_beam=False
    )
    npt.assert_almost_equal(
        source_without_pb_image1, source_without_pb_image2, decimal=8
    )

    ## test "apply_primary_beam" of lens_surface_brightness function
    lens_light_without_pb_image1 = (
        imageModel_with_psf_without_pb.lens_surface_brightness(
            kwargs_lens_light, unconvolved=True
        )
    )
    lens_light_without_pb_image2 = imageModel_with_pb_psf.lens_surface_brightness(
        kwargs_lens_light, unconvolved=True, apply_primary_beam=False
    )
    npt.assert_almost_equal(
        lens_light_without_pb_image1, lens_light_without_pb_image2, decimal=8
    )

    ## test "apply_primary_beam" of image function
    image_without_pb_image1 = imageModel_with_psf_without_pb.image(
        kwargs_lens, kwargs_source, kwargs_lens_light, unconvolved=True
    )
    image_without_pb_image2 = imageModel_with_pb_psf.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        unconvolved=True,
        apply_primary_beam=False,
    )
    npt.assert_almost_equal(image_without_pb_image1, image_without_pb_image2, decimal=8)


def test_point_source_simulation_with_interferometric_PSF_and_primary_beam():
    sigma_bkg = 0.05
    exp_time = np.inf
    numPix = 100
    deltaPix = 0.2

    # simulate a primary beam (pb)
    primary_beam = np.zeros((numPix, numPix))
    for i in range(numPix):
        for j in range(numPix):
            primary_beam[i, j] = np.exp(-2e-4 * ((i - 78) ** 2 + (j - 56) ** 2))
    primary_beam /= np.max(primary_beam)

    # simulate a spherical sinc function as psf, which contains negative pixels
    psf_test = np.zeros((221, 221))
    for i in range(221):
        for j in range(221):
            if i > j:
                psf_test[i, j] = psf_test[j, i]
            r = np.sqrt((i - 110) ** 2 + (j - 110) ** 2)
            if r == 0:
                psf_test[i, j] = 1
            else:
                psf_test[i, j] = np.sin(r * 0.5) / (r * 0.5)

    # Define data class with the primary beam
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
    kwargs_data["ra_at_xy_0"] = -(50) * deltaPix
    kwargs_data["dec_at_xy_0"] = -(50) * deltaPix
    kwargs_data["antenna_primary_beam"] = primary_beam
    data_class = ImageData(**kwargs_data)

    # Define two PSF classes
    kwargs_psf_none = {"psf_type": "NONE", "pixel_size": deltaPix}
    psf_class_none = PSF(**kwargs_psf_none)

    kernel_cut = kernel_util.cut_psf(psf_test, 201, normalisation=False)
    kwargs_psf = {
        "psf_type": "PIXEL",
        "pixel_size": deltaPix,
        "kernel_point_source": kernel_cut,
        "kernel_point_source_normalisation": False,
    }
    psf_class = PSF(**kwargs_psf)

    # define lens model
    kwargs_lens = [
        {
            "theta_E": 1.7,
            "gamma": 2.0,
            "e1": -0.02,
            "e2": -0.01,
            "center_x": -0.31,
            "center_y": -0.26,
        }
    ]
    lens_model_list = ["EPL"]
    lens_model_class = LensModel(lens_model_list=lens_model_list)

    # Define the Point Source class
    point_source_model_list = ["UNLENSED", "LENSED_POSITION", "SOURCE_POSITION"]
    pointSource = PointSource(
        point_source_type_list=point_source_model_list,
        lens_model=lens_model_class,
        fixed_magnification_list=[True, True, True],
    )
    kwargs_ps = [
        {
            "ra_image": [-1.7, -0.4, -1.57],
            "dec_image": [0.21, 0.99, 2.21],
            "point_amp": [100, 100, 100],
        },
        {
            "ra_image": [1.49, 3.85, 0.81],
            "dec_image": [2.51, 1.67, -3.78],
            "source_amp": [100, 100, 100],
        },
        {"ra_source": 3.23, "dec_source": -0.40, "source_amp": 100},
    ]

    # Define the image model class
    imageModel_no_psf = ImageModel(
        data_class,
        psf_class_none,
        lens_model_class=lens_model_class,
        point_source_class=pointSource,
    )

    imageModel_with_psf = ImageModel(
        data_class,
        psf_class,
        lens_model_class=lens_model_class,
        point_source_class=pointSource,
    )

    # Generate images with different psf, pb conditions
    image_uncon_no_pb = imageModel_no_psf.image(
        kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, apply_primary_beam=False
    )
    image_uncon = imageModel_no_psf.image(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps)
    image_no_pb = imageModel_with_psf.image(
        kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, apply_primary_beam=False
    )
    image = imageModel_with_psf.image(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps)

    # test the values by testing the summations
    npt.assert_almost_equal(np.sum(image_uncon_no_pb), 998.0643356369997, decimal=8)
    npt.assert_almost_equal(np.sum(image_uncon), 834.2233081496132, decimal=8)
    npt.assert_almost_equal(np.sum(image_no_pb), 25539.14307941169, decimal=8)
    npt.assert_almost_equal(np.sum(image), 21228.25308383333, decimal=8)
