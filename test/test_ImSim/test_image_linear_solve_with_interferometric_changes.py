__author__ = "nan zhang"

import numpy as np
import numpy.testing as npt
import scipy.signal
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Util import kernel_util
import lenstronomy.Util.util as util

from lenstronomy.ImSim.image_linear_solve import ImageLinearFit

"""
Test the linear solver for natwt (natural weighting) interferometric data.
Test the _image_linear_solve function of ImageLinearFit class.
The idea is to define data, psf, source, lens, lens light classes respectively, and run the linear solving
inside and outside of the _image_linear_solve function. Verify the 1st and 4th output of _image_linear_solve.
The test should be independent of the specific definitions of the light and lens profiles.
"""


def test_image_linear_solve_with_primary_beam_and_interferometry_psf():
    background_rms = 3.0
    exp_time = np.inf
    numPix = 80
    deltaPix = 0.05
    psf_type = "PIXEL"
    kernel_size = 161

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

    kwargs_data = sim_util.data_configure_simple(
        numPix, deltaPix, exp_time, background_rms
    )
    kwargs_data["ra_at_xy_0"] = -(40) * deltaPix
    kwargs_data["dec_at_xy_0"] = -(40) * deltaPix
    kwargs_data["antenna_primary_beam"] = primary_beam
    kwargs_data["likelihood_method"] = (
        "interferometry_natwt"  # testing just for interferometry natwt method
    )
    data_class = ImageData(**kwargs_data)

    kernel_cut = kernel_util.cut_psf(psf_test, kernel_size, normalisation=False)
    kwargs_psf = {
        "psf_type": psf_type,
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
        "amp": 25.0,
        "R_sersic": 0.3,
        "n_sersic": 2,
        "center_x": 0,
        "center_y": 0,
    }
    lens_light_model_list = ["SERSIC"]
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

    kwargs_sersic_ellipse = {
        "amp": 10.0,
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

    imageModel = ImageModel(
        data_class,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        kwargs_numerics=kwargs_numerics,
    )
    image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light)

    # note that the simulated noise here is not the interferometric noise. we just use it to test the numerics
    np.random.seed(42)
    test_noise = scipy.signal.fftconvolve(
        np.random.normal(0, 1, (numPix, numPix)), psf_test, mode="same"
    )
    test_noise *= background_rms / np.std(test_noise)
    sim_data = image_sim + test_noise
    data_class.update_data(sim_data)

    # define the ImageLinearFit class using the materials defined above, run the image_linear_solve function
    imageLinearFit = ImageLinearFit(
        data_class,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        kwargs_numerics=kwargs_numerics,
    )
    model, _, param_cov, amps = imageLinearFit.image_linear_solve(
        kwargs_lens, kwargs_source, kwargs_lens_light
    )

    # execute the same linear solving outside of the image_linear_solve function
    A = imageLinearFit.linear_response_matrix_interferometry_unconvolved(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps=None
    )
    A0 = util.array2image(A[0])
    A1 = util.array2image(A[1])
    A0c = scipy.signal.fftconvolve(A0, psf_test, mode="same")
    A1c = scipy.signal.fftconvolve(A1, psf_test, mode="same")
    M = np.zeros((2, 2))
    b = np.zeros((2))
    M[0, 0] = np.sum(A0c * A0)
    M[0, 1] = np.sum(A0c * A1)
    M[1, 0] = np.sum(A1c * A0)
    M[1, 1] = np.sum(A1c * A1)
    b[0] = np.sum(A0 * sim_data)
    b[1] = np.sum(A1 * sim_data)

    M /= background_rms**2
    b /= background_rms**2

    amps0 = np.linalg.lstsq(M, b, rcond=None)[0]
    unconvolved_model = amps0[0] * A0 + amps0[1] * A1
    dirty_model = amps0[0] * A0c + amps0[1] * A1c

    npt.assert_almost_equal([unconvolved_model, dirty_model], model, decimal=8)
    npt.assert_almost_equal(amps0, amps, decimal=8)
    assert param_cov is None

    # test param_cov
    model_1, _, param_cov_1, amps_1 = imageLinearFit.image_linear_solve(
        kwargs_lens, kwargs_source, kwargs_lens_light, inv_bool=True
    )
    param_cov_1_expected = np.linalg.inv(M)
    npt.assert_almost_equal([unconvolved_model, dirty_model], model_1, decimal=8)
    npt.assert_almost_equal(amps0, amps_1, decimal=8)
    npt.assert_almost_equal(param_cov_1_expected, param_cov_1, decimal=8)

    # test the output of linear_response_matrix_interferometry_unconvolved with only
    # source light and lens light (so without the point sources) being the same with the
    # output of linear_response_matrix in the unconvolved condition times primary beam
    A_default_unconvolve = imageLinearFit.linear_response_matrix(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps=None, unconvolved=True
    )
    A_default_unconvolve *= primary_beam.reshape(6400)
    npt.assert_almost_equal(A_default_unconvolve, A, decimal=8)


def test_interferometry_image_linear_solve_with_point_source():
    background_rms = 3.0
    exp_time = np.inf
    numPix = 80
    deltaPix = 0.05
    psf_type = "PIXEL"
    kernel_size = 161

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

    kwargs_data = sim_util.data_configure_simple(
        numPix, deltaPix, exp_time, background_rms
    )
    kwargs_data["ra_at_xy_0"] = -(40) * deltaPix
    kwargs_data["dec_at_xy_0"] = -(40) * deltaPix
    kwargs_data["antenna_primary_beam"] = primary_beam
    kwargs_data["likelihood_method"] = (
        "interferometry_natwt"  # testing just for interferometry natwt method
    )
    data_class = ImageData(**kwargs_data)

    kernel_cut = kernel_util.cut_psf(psf_test, kernel_size, normalisation=False)
    kwargs_psf = {
        "psf_type": psf_type,
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
        "center_x": 0.0,
        "center_y": 0.0,
        "e1": 0.1,
        "e2": 0.04,
    }
    lens_model_list = ["SPEP", "SHEAR"]
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)

    kwargs_sersic = {
        "amp": 25.0,
        "R_sersic": 0.3,
        "n_sersic": 2,
        "center_x": 0,
        "center_y": 0,
    }
    lens_light_model_list = ["SERSIC"]
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

    kwargs_sersic_ellipse = {
        "amp": 10.0,
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

    kwargs_ps = [
        {"ra_image": [0.23, 0.34], "dec_image": [0.14, -0.12], "point_amp": [10, 5]},
        {"ra_image": [-0.13, -0.26], "dec_image": [0.12, 0.29], "point_amp": [6, 20]},
        {"ra_image": [-0.35, 0.06], "dec_image": [0.12, 0.29], "source_amp": [15]},
        {"ra_source": 0.1, "dec_source": -0.4, "point_amp": [30]},
        {"ra_source": -0.7, "dec_source": 0.5, "source_amp": [30]},
    ]
    point_source_model_list = [
        "UNLENSED",
        "LENSED_POSITION",
        "LENSED_POSITION",
        "SOURCE_POSITION",
        "SOURCE_POSITION",
    ]
    pointSource = PointSource(
        point_source_type_list=point_source_model_list,
        lens_model=lens_model_class,
        fixed_magnification_list=[False, False, True, False, True],
    )

    kwargs_numerics = {"supersampling_factor": 1, "supersampling_convolution": False}

    imageModel = ImageModel(
        data_class,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics,
    )
    image_sim = imageModel.image(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    )

    # note that the simulated noise here is not the interferometric noise. we just use it to test the numerics
    np.random.seed(42)
    test_noise = scipy.signal.fftconvolve(
        np.random.normal(0, 1, (numPix, numPix)), psf_test, mode="same"
    )
    test_noise *= background_rms / np.std(test_noise)
    sim_data = image_sim + test_noise
    data_class.update_data(sim_data)

    # define the ImageLinearFit class using the materials defined above, run the image_linear_solve function
    imageLinearFit = ImageLinearFit(
        data_class,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics,
    )
    model, _, param_cov, amps = imageLinearFit.image_linear_solve(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    )

    # execute the same linear solving outside of the image_linear_solve function
    A = imageLinearFit.linear_response_matrix_interferometry_unconvolved(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    )
    Ns = A.shape[0]
    A = A.reshape(Ns, 80, 80)
    A_convolved = np.zeros(A.shape)
    for i in range(Ns):
        A_convolved[i] = scipy.signal.fftconvolve(A[i], psf_test, mode="same")
    M = np.zeros((Ns, Ns))
    b = np.zeros((Ns))
    for i in range(Ns):
        for j in range(Ns):
            M[i, j] = np.sum(A_convolved[i] * A[j])
        b[i] = np.sum(A[i] * sim_data)

    M /= background_rms**2
    b /= background_rms**2

    amps0 = np.linalg.lstsq(M, b, rcond=None)[0]
    unconvolved_model = A.reshape(Ns, 6400).T.dot(amps0).reshape(80, 80)
    dirty_model = A_convolved.reshape(Ns, 6400).T.dot(amps0).reshape(80, 80)

    npt.assert_almost_equal([unconvolved_model, dirty_model], model, decimal=8)
    npt.assert_almost_equal(amps0, amps, decimal=8)
    assert param_cov is None

    # test inv_bool=True
    model_1, _, param_cov_1, amps_1 = imageLinearFit.image_linear_solve(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, inv_bool=True
    )
    param_cov_1_expected = np.linalg.inv(M)
    npt.assert_almost_equal(model_1, model, decimal=8)
    npt.assert_almost_equal(amps_1, amps, decimal=8)
    npt.assert_almost_equal(param_cov_1, param_cov_1_expected, decimal=8)


def test_interferometry_image_linear_solve_with_point_source_without_pb_input():
    background_rms = 3.0
    exp_time = np.inf
    numPix = 80
    deltaPix = 0.05
    psf_type = "PIXEL"
    kernel_size = 161

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

    kwargs_data = sim_util.data_configure_simple(
        numPix, deltaPix, exp_time, background_rms
    )
    kwargs_data["ra_at_xy_0"] = -(40) * deltaPix
    kwargs_data["dec_at_xy_0"] = -(40) * deltaPix
    kwargs_data["likelihood_method"] = (
        "interferometry_natwt"  # testing just for interferometry natwt method
    )
    data_class = ImageData(**kwargs_data)

    kernel_cut = kernel_util.cut_psf(psf_test, kernel_size, normalisation=False)
    kwargs_psf = {
        "psf_type": psf_type,
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
        "center_x": 0.0,
        "center_y": 0.0,
        "e1": 0.1,
        "e2": 0.04,
    }
    lens_model_list = ["SPEP", "SHEAR"]
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)

    kwargs_sersic = {
        "amp": 25.0,
        "R_sersic": 0.3,
        "n_sersic": 2,
        "center_x": 0,
        "center_y": 0,
    }
    lens_light_model_list = ["SERSIC"]
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

    kwargs_sersic_ellipse = {
        "amp": 10.0,
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

    kwargs_ps = [
        {"ra_image": [0.23, 0.34], "dec_image": [0.14, -0.12], "point_amp": [10, 5]},
        {"ra_image": [-0.13, -0.26], "dec_image": [0.12, 0.29], "point_amp": [6, 20]},
        {"ra_image": [-0.35, 0.06], "dec_image": [0.12, 0.29], "source_amp": [15]},
        {"ra_source": 0.1, "dec_source": -0.4, "point_amp": [30]},
        {"ra_source": -0.7, "dec_source": 0.5, "source_amp": [30]},
    ]
    point_source_model_list = [
        "UNLENSED",
        "LENSED_POSITION",
        "LENSED_POSITION",
        "SOURCE_POSITION",
        "SOURCE_POSITION",
    ]
    pointSource = PointSource(
        point_source_type_list=point_source_model_list,
        lens_model=lens_model_class,
        fixed_magnification_list=[False, False, True, False, True],
    )

    kwargs_numerics = {"supersampling_factor": 1, "supersampling_convolution": False}

    imageModel = ImageModel(
        data_class,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics,
    )
    image_sim = imageModel.image(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    )

    # note that the simulated noise here is not the interferometric noise. we just use it to test the numerics
    np.random.seed(42)
    test_noise = scipy.signal.fftconvolve(
        np.random.normal(0, 1, (numPix, numPix)), psf_test, mode="same"
    )
    test_noise *= background_rms / np.std(test_noise)
    sim_data = image_sim + test_noise
    data_class.update_data(sim_data)

    # define the ImageLinearFit class using the materials defined above, run the image_linear_solve function
    imageLinearFit = ImageLinearFit(
        data_class,
        psf_class,
        lens_model_class,
        source_model_class,
        lens_light_model_class,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics,
    )
    model, _, param_cov, amps = imageLinearFit.image_linear_solve(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    )

    # execute the same linear solving outside of the image_linear_solve function
    A = imageLinearFit.linear_response_matrix_interferometry_unconvolved(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps
    )
    Ns = A.shape[0]
    A = A.reshape(Ns, 80, 80)
    A_convolved = np.zeros(A.shape)
    for i in range(Ns):
        A_convolved[i] = scipy.signal.fftconvolve(A[i], psf_test, mode="same")
    M = np.zeros((Ns, Ns))
    b = np.zeros((Ns))
    for i in range(Ns):
        for j in range(Ns):
            M[i, j] = np.sum(A_convolved[i] * A[j])
        b[i] = np.sum(A[i] * sim_data)

    M /= background_rms**2
    b /= background_rms**2

    amps0 = np.linalg.lstsq(M, b, rcond=None)[0]
    unconvolved_model = A.reshape(Ns, 6400).T.dot(amps0).reshape(80, 80)
    dirty_model = A_convolved.reshape(Ns, 6400).T.dot(amps0).reshape(80, 80)

    npt.assert_almost_equal([unconvolved_model, dirty_model], model, decimal=8)
    npt.assert_almost_equal(amps0, amps, decimal=8)
    assert param_cov is None

    # test inv_bool=True
    model_1, _, param_cov_1, amps_1 = imageLinearFit.image_linear_solve(
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, inv_bool=True
    )
    param_cov_1_expected = np.linalg.inv(M)
    npt.assert_almost_equal(model_1, model, decimal=8)
    npt.assert_almost_equal(amps_1, amps, decimal=8)
    npt.assert_almost_equal(param_cov_1, param_cov_1_expected, decimal=8)
