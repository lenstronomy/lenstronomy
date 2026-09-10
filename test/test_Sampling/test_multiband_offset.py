import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose

from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Util import util
from lenstronomy.Sampling.Likelihoods.image_likelihood import ImageLikelihood


# Generate simple two-band mock data
def make_mock_multiband_data():
    num_pix = 40
    delta_pix = 0.1

    background_rms = 0.01
    exposure_time = 1000

    psf_kwargs = {
        "psf_type": "GAUSSIAN",
        "fwhm": 0.1,
        "pixel_size": delta_pix,
    }

    numerics_kwargs = {
        "supersampling_factor": 1,
        "supersampling_convolution": False,
    }

    # same coordinate system initially
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = (
        util.make_grid_with_coordtransform(
            num_pix=num_pix,
            delta_pix=delta_pix,
            center_ra=0,
            center_dec=0,
            subgrid_res=1,
            inverse=False,
        )
    )

    kwargs_data_1 = {
        "image_data": np.zeros((num_pix, num_pix)),
        "background_rms": background_rms,
        "exposure_time": exposure_time,
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": Mpix2coord,
    }

    # band 2 starts from identical coordinate system
    kwargs_data_2 = copy.deepcopy(kwargs_data_1)

    multi_band_list = [
        [
            kwargs_data_1,
            psf_kwargs,
            numerics_kwargs,
        ],
        [
            kwargs_data_2,
            psf_kwargs,
            numerics_kwargs,
        ],
    ]

    # --------------------------------------------------------
    # lens model
    # --------------------------------------------------------

    lens_model_list = [
        "SIE",
    ]

    kwargs_lens = [
        {
            "theta_E": 1.0,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.05,
            "e2": 0.05,
        }
    ]

    # --------------------------------------------------------
    # source
    # --------------------------------------------------------
    source_model_list = [
        "SERSIC_ELLIPSE",
        "SERSIC_ELLIPSE",
    ]

    kwargs_source = [
        {
            "amp": 10,
            "R_sersic": 0.2,
            "n_sersic": 2,
            "e1": 0,
            "e2": 0,
            "center_x": 0,
            "center_y": 0,
        },
        {
            "amp": 5,
            "R_sersic": 0.2,
            "n_sersic": 2,
            "e1": 0,
            "e2": 0,
            "center_x": 0,
            "center_y": 0,
        },
    ]

    kwargs_model = {
        "lens_model_list": lens_model_list,
        "source_light_model_list": source_model_list,
        "lens_light_model_list": [],
        "index_source_light_model_list": [
            [0],
            [1],
        ],
    }

    # generate images
    sim_band_1 = SingleBandMultiModel(
        multi_band_list=multi_band_list,
        kwargs_model=kwargs_model,
        likelihood_mask_list=None,
        band_index=0,
    )
    sim_band_2 = SingleBandMultiModel(
        multi_band_list=multi_band_list,
        kwargs_model=kwargs_model,
        likelihood_mask_list=None,
        band_index=1,
    )

    image_1 = sim_band_1.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
    )
    image_2 = sim_band_2.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
    )

    kwargs_data_1["image_data"] = image_1
    kwargs_data_2["image_data"] = image_2

    kwargs_data_joint = {
        "multi_band_list": multi_band_list,
        "multi_band_type": "multi-linear",
    }

    return (
        kwargs_data_joint,
        kwargs_model,
        kwargs_lens,
        kwargs_source,
        [],
    )


def make_kwargs_special(
    ra_shift=0.0,
    dec_shift=0.0,
    phi_rot=0.0,
):
    return {
        "kwargs_offsets": [
            {},
            {
                "ra_shift": ra_shift,
                "dec_shift": dec_shift,
                "phi_rot": phi_rot,
            },
        ]
    }


def make_likelihood():

    (
        kwargs_data_joint,
        kwargs_model,
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
    ) = make_mock_multiband_data()

    kwargs_likelihood = {
        "source_marg": False,
    }

    likelihood = ImageLikelihood(
        multi_band_list=kwargs_data_joint["multi_band_list"],
        multi_band_type=kwargs_data_joint["multi_band_type"],
        kwargs_model=kwargs_model,
        **kwargs_likelihood,
    )

    return (
        likelihood,
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
    )


# Test coordinate updates through the ImageModel API
def test_multiband_offsets_change_image():
    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()

    image_model = likelihood.imSim._image_model_list[1]

    kwargs_special_zero = make_kwargs_special()

    kwargs_special_offset = make_kwargs_special(
        ra_shift=0.05,
        dec_shift=-0.03,
        phi_rot=0.02,
    )

    image_zero = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    image_offset = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_offset,
    )

    assert np.max(np.abs(image_offset - image_zero)) > 0


def test_multiband_offsets_restore_reference_image():
    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()

    image_model = likelihood.imSim._image_model_list[1]

    kwargs_special_zero = make_kwargs_special()

    kwargs_special_offset = make_kwargs_special(
        ra_shift=0.05,
        dec_shift=-0.03,
        phi_rot=0.02,
    )

    image_zero = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    image_offset = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_offset,
    )

    image_restored = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    assert np.max(np.abs(image_offset - image_zero)) > 0
    assert_allclose(image_restored, image_zero)


def test_multiband_offsets_restore_reference_coordinates():
    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()

    image_model = likelihood.imSim._image_model_list[1]

    kwargs_special_zero = make_kwargs_special()

    kwargs_special_offset = make_kwargs_special(
        ra_shift=0.05,
        dec_shift=-0.03,
        phi_rot=0.02,
    )

    image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    ra_zero, dec_zero = image_model.ImageNumerics.coordinates_evaluate
    ra_zero = np.copy(ra_zero)
    dec_zero = np.copy(dec_zero)

    image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_offset,
    )

    ra_offset, dec_offset = image_model.ImageNumerics.coordinates_evaluate

    assert not np.allclose(ra_offset, ra_zero)
    assert not np.allclose(dec_offset, dec_zero)

    image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    ra_restored, dec_restored = image_model.ImageNumerics.coordinates_evaluate

    assert_allclose(ra_restored, ra_zero)
    assert_allclose(dec_restored, dec_zero)


def test_repeated_multiband_offsets_are_consistent():
    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()

    image_model = likelihood.imSim._image_model_list[1]

    kwargs_special_zero = make_kwargs_special()

    kwargs_special_offset = make_kwargs_special(
        ra_shift=0.05,
        dec_shift=-0.03,
        phi_rot=0.02,
    )

    # First application
    image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    image_offset_1 = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_offset,
    )

    # Return to reference frame
    image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    # Apply the same offset again
    image_offset_2 = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_offset,
    )

    assert_allclose(image_offset_1, image_offset_2)


# Test likelihood response to offsets
def test_multiband_offsets_change_likelihood():
    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()

    kwargs_special_zero = make_kwargs_special()

    kwargs_special_offset = make_kwargs_special(
        ra_shift=0.05,
        dec_shift=-0.03,
        phi_rot=0.02,
    )

    logL_zero, _ = likelihood.logL(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    logL_offset, _ = likelihood.logL(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps=None,
        kwargs_special=kwargs_special_offset,
    )

    assert np.isfinite(logL_zero)
    assert np.isfinite(logL_offset)
    assert logL_zero != logL_offset


def test_correct_multiband_offsets_improve_likelihood():
    (
        kwargs_data_joint,
        kwargs_model,
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
    ) = make_mock_multiband_data()

    true_offsets = {
        "ra_shift": 0.05,
        "dec_shift": -0.03,
        "phi_rot": 0.02,
    }

    # Generate band 2 with a known coordinate offset.

    multi_band_list_offset = copy.deepcopy(kwargs_data_joint["multi_band_list"])

    multi_band_list_offset[1][0].update(true_offsets)

    sim_band_2 = SingleBandMultiModel(
        multi_band_list=multi_band_list_offset,
        kwargs_model=kwargs_model,
        likelihood_mask_list=None,
        band_index=1,
    )

    image_2_offset = sim_band_2.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
    )

    # Use the offset image as the observed data with the nominal
    # coordinate frame for fitting.
    kwargs_data_joint["multi_band_list"][1][0]["image_data"] = image_2_offset

    likelihood = ImageLikelihood(
        multi_band_list=kwargs_data_joint["multi_band_list"],
        multi_band_type=kwargs_data_joint["multi_band_type"],
        kwargs_model=kwargs_model,
        source_marg=False,
    )

    kwargs_special_zero = make_kwargs_special()

    kwargs_special_true = {
        "kwargs_offsets": [
            {},
            true_offsets,
        ]
    }

    kwargs_special_wrong = {
        "kwargs_offsets": [
            {},
            {
                "ra_shift": -true_offsets["ra_shift"],
                "dec_shift": -true_offsets["dec_shift"],
                "phi_rot": -true_offsets["phi_rot"],
            },
        ]
    }

    logL_zero, _ = likelihood.logL(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    logL_true, _ = likelihood.logL(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps=None,
        kwargs_special=kwargs_special_true,
    )

    logL_wrong, _ = likelihood.logL(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps=None,
        kwargs_special=kwargs_special_wrong,
    )

    # Verify that the same-sign offset used to generate the mock data
    # gives a higher likelihood than both zero and the opposite offset.
    assert logL_true > logL_zero
    assert logL_true > logL_wrong


if __name__ == "__main__":
    pytest.main()
