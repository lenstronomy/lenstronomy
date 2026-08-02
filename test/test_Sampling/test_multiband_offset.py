import numpy as np
from numpy.testing import assert_almost_equal

# Same transformation logic as implemented in ImageLikelihood
def apply_coordinate_transformation(
    ra_0_old,
    dec_0_old,
    M_old,
    nx,
    ny,
    dx,
    dy,
    angle,
):
    """
    Apply rotation around the geometric center and translation.

    This reproduces the coordinate transformation implemented
    in the multi-band offset feature.
    """

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    # Compute the original geometric center
    ra_center = (
        ra_0_old
        + M_old[0, 0] * cx
        + M_old[0, 1] * cy
    )

    dec_center = (
        dec_0_old
        + M_old[1, 0] * cx
        + M_old[1, 1] * cy
    )

    # Apply rotation
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    rot_matrix = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ]
    )

    M_new = np.dot(rot_matrix, M_old)

    # Keep the geometric center fixed during rotation,
    # then apply the requested translation
    ra_0_new = (
        ra_center
        - (M_new[0, 0] * cx + M_new[0, 1] * cy)
        + dx
    )

    dec_0_new = (
        dec_center
        - (M_new[1, 0] * cx + M_new[1, 1] * cy)
        + dy
    )

    return ra_0_new, dec_0_new, M_new


def test_transformation():
    """
    Test combined rotation and translation.
    """

    # Set up a 10x10 grid with pixel scale of 0.1
    nx, ny = 10, 10

    M_old = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )

    # Initial coordinate origin.
    # The geometric center is located at (0, 0).
    ra_0_old = -0.45
    dec_0_old = -0.45

    # Apply translation and 90-degree rotation
    dx, dy = 0.2, 0.1
    angle = np.pi / 2

    ra_0_new, dec_0_new, M_new = apply_coordinate_transformation(
        ra_0_old,
        dec_0_old,
        M_old,
        nx,
        ny,
        dx,
        dy,
        angle,
    )

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    # Verify that the new center only changes due to translation
    new_ra_center = (
        ra_0_new
        + M_new[0, 0] * cx
        + M_new[0, 1] * cy
    )

    new_dec_center = (
        dec_0_new
        + M_new[1, 0] * cx
        + M_new[1, 1] * cy
    )

    assert_almost_equal(new_ra_center, dx)
    assert_almost_equal(new_dec_center, dy)

    # Verify that the transformation matrix is rotated correctly
    assert_almost_equal(M_new[0, 0], 0.0)
    assert_almost_equal(M_new[1, 0], 0.1)

    print("Test passed: rotation around the center and translation.")


def test_translation_only():

    nx, ny = 10, 10

    M_old = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )

    ra_0_old = -0.45
    dec_0_old = -0.45

    dx = 0.2
    dy = -0.1
    angle = 0.0

    ra_0_new, dec_0_new, M_new = apply_coordinate_transformation(
        ra_0_old,
        dec_0_old,
        M_old,
        nx,
        ny,
        dx,
        dy,
        angle,
    )

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    # Check the updated coordinate center
    ra_center_new = (
        ra_0_new
        + M_new[0, 0] * cx
        + M_new[0, 1] * cy
    )

    dec_center_new = (
        dec_0_new
        + M_new[1, 0] * cx
        + M_new[1, 1] * cy
    )

    assert_almost_equal(ra_center_new, dx)
    assert_almost_equal(dec_center_new, dy)

    # Translation should not modify the transformation matrix
    assert_almost_equal(M_new, M_old)

    print("Test passed: translation only.")


def test_rotation_only():

    nx, ny = 10, 10

    M_old = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )

    ra_0_old = -0.45
    dec_0_old = -0.45

    dx = 0.0
    dy = 0.0
    angle = np.pi / 2

    ra_0_new, dec_0_new, M_new = apply_coordinate_transformation(
        ra_0_old,
        dec_0_old,
        M_old,
        nx,
        ny,
        dx,
        dy,
        angle,
    )

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    # Original center
    ra_center_old = (
        ra_0_old
        + M_old[0, 0] * cx
        + M_old[0, 1] * cy
    )

    dec_center_old = (
        dec_0_old
        + M_old[1, 0] * cx
        + M_old[1, 1] * cy
    )

    # New center after rotation
    ra_center_new = (
        ra_0_new
        + M_new[0, 0] * cx
        + M_new[0, 1] * cy
    )

    dec_center_new = (
        dec_0_new
        + M_new[1, 0] * cx
        + M_new[1, 1] * cy
    )

    # Rotation around the center should preserve the center position
    assert_almost_equal(ra_center_new, ra_center_old)
    assert_almost_equal(dec_center_new, dec_center_old)

    # Verify axis rotation
    assert_almost_equal(M_new[0, 0], 0.0)
    assert_almost_equal(M_new[1, 0], 0.1)

    print("Test passed: rotation only.")


def test_shift_and_rotation():

    nx, ny = 10, 10

    M_old = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )

    ra_0_old = -0.45
    dec_0_old = -0.45

    dx = 0.2
    dy = 0.1
    angle = np.pi / 2

    ra_0_new, dec_0_new, M_new = apply_coordinate_transformation(
        ra_0_old,
        dec_0_old,
        M_old,
        nx,
        ny,
        dx,
        dy,
        angle,
    )

    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

    ra_center_new = (
        ra_0_new
        + M_new[0, 0] * cx
        + M_new[0, 1] * cy
    )

    dec_center_new = (
        dec_0_new
        + M_new[1, 0] * cx
        + M_new[1, 1] * cy
    )

    # The final center should match the applied translation
    assert_almost_equal(ra_center_new, dx)
    assert_almost_equal(dec_center_new, dy)

    print("Test passed: combined shift and rotation.")


def test_center_preservation_without_shift():

    nx, ny = 20, 15

    M_old = np.array(
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )

    ra_0_old = -0.95
    dec_0_old = -0.70

    dx = 0
    dy = 0
    angle = 0.3

    ra_0_new, dec_0_new, M_new = apply_coordinate_transformation(
        ra_0_old,
        dec_0_old,
        M_old,
        nx,
        ny,
        dx,
        dy,
        angle,
    )

    cx, cy = (nx - 1)/2, (ny - 1)/2

    old_center = np.array([
        ra_0_old + M_old[0,0]*cx + M_old[0,1]*cy,
        dec_0_old + M_old[1,0]*cx + M_old[1,1]*cy,
    ])

    new_center = np.array([
        ra_0_new + M_new[0,0]*cx + M_new[0,1]*cy,
        dec_0_new + M_new[1,0]*cx + M_new[1,1]*cy,
    ])

    assert_almost_equal(old_center, new_center)

    print("Test passed: rotation preserves geometric center.")

if __name__ == "__main__":

    test_transformation()
    test_translation_only()
    test_rotation_only()
    test_shift_and_rotation()
    test_center_preservation_without_shift()

# ============================================================

import copy
import numpy as np
from numpy.testing import assert_allclose

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import util
from lenstronomy.Sampling.Likelihoods.image_likelihood import ImageLikelihood

# ============================================================
# 1. Generate simple two-band mock data
# ============================================================

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
            numPix=num_pix,
            deltapix=delta_pix,
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
        [kwargs_data_1,
            psf_kwargs,
            numerics_kwargs,],
        [kwargs_data_2,
            psf_kwargs,
            numerics_kwargs,],]

    # --------------------------------------------------------
    # lens model
    # --------------------------------------------------------

    lens_model_list = [
        "SIE",
    ]

    lens_model = LensModel(
        lens_model_list=lens_model_list
    )

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

    source_model = LightModel(
        light_model_list=source_model_list
    )

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


# ============================================================
# 2. Build likelihood
# ============================================================

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

# ============================================================
# 3. Test apply / restore
# ============================================================

def test_apply_restore_multiband_offsets():

    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()
    image_model_list = (
        likelihood.imSim._imageModel_list
    )
    data_band_1 = (
        image_model_list[1].Data
    )

    original_ra = data_band_1._ra_at_xy_0
    original_dec = data_band_1._dec_at_xy_0
    original_M = np.copy(
        data_band_1._Mpix2a
    )

    kwargs_special = {
        "kwargs_offsets": [
            {},
            {
                "dx": 0.05,
                "dy": -0.03,
                "angle": 0.02,
            },

        ]

    }
    # --------------------------------------------------------
    # Apply
    # --------------------------------------------------------
    backup = (
        likelihood._apply_multiband_offsets(
            kwargs_special
        )
    )

    assert not np.isclose(
        data_band_1._ra_at_xy_0,
        original_ra,
    )

    assert not np.isclose(
        data_band_1._dec_at_xy_0,
        original_dec,
    )

    assert not np.allclose(
        data_band_1._Mpix2a,
        original_M,
    )

    print(
        "PASS: apply_multiband_offsets modifies coordinates"
    )

    # --------------------------------------------------------
    # Restore
    # --------------------------------------------------------

    likelihood._restore_multiband_offsets(
        backup
    )


    assert_allclose(
        data_band_1._ra_at_xy_0,
        original_ra,
    )

    assert_allclose(
        data_band_1._dec_at_xy_0,
        original_dec,
    )

    assert_allclose(
        data_band_1._Mpix2a,
        original_M,
    )

    print(
        "PASS: restore_multiband_offsets restores coordinates"
    )

# ============================================================
# 4. Test image response to multiband offsets
# ============================================================
def test_multiband_offsets_change_image_and_restore():

    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()

    image_model = likelihood.imSim._imageModel_list[1]

    kwargs_lens = [
        {
            "theta_E": 1.0,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.05,
            "e2": 0.05,
        }
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


    kwargs_special_zero = {
        "kwargs_offsets": [
            {},
            {
                "dx": 0,
                "dy": 0,
                "angle": 0,
            },
        ]
    }


    kwargs_special_true = {
        "kwargs_offsets": [
            {},
            {
                "dx": 0.05,
                "dy": -0.03,
                "angle": 0.02,
            },
        ]
    }


    # --------------------------------------------------------
    # image without offset
    # --------------------------------------------------------

    image_zero = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
        kwargs_special=kwargs_special_zero,
    )

    # --------------------------------------------------------
    # image with offset
    # --------------------------------------------------------

    backup = likelihood._apply_multiband_offsets(
        kwargs_special_true
    )

    image_true = image_model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=[],
        kwargs_ps=None,
    )

    likelihood._restore_multiband_offsets(
        backup
    )

    # image should change
    assert np.max(np.abs(image_true-image_zero)) > 0

    # --------------------------------------------------------
    # coordinate should change
    # --------------------------------------------------------

    ra0, dec0 = image_model.ImageNumerics.coordinates_evaluate

    backup = likelihood._apply_multiband_offsets(
        kwargs_special_true
    )

    ra1, dec1 = image_model.ImageNumerics.coordinates_evaluate

    assert np.max(np.abs(ra1-ra0)) > 0
    assert np.max(np.abs(dec1-dec0)) > 0


    likelihood._restore_multiband_offsets(
        backup
    )


    ra2, dec2 = image_model.ImageNumerics.coordinates_evaluate

    assert_allclose(ra2, ra0)
    assert_allclose(dec2, dec0)

    print(
        "PASS: multiband offsets change image and restore correctly"
    )

def test_repeated_apply_gives_same_image():

    likelihood, kwargs_lens, kwargs_source, kwargs_lens_light = make_likelihood()

    model = likelihood.imSim._imageModel_list[1]

    kwargs_special = {
        "kwargs_offsets":[
            {},
            {
                "dx":0.05,
                "dy":-0.03,
                "angle":0.02,
            }
        ]
    }

    backup = likelihood._apply_multiband_offsets(
        kwargs_special
    )

    image1 = model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps=None,
    )

    likelihood._restore_multiband_offsets(
        backup
    )

    backup = likelihood._apply_multiband_offsets(
        kwargs_special
    )

    image2 = model.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps=None,
    )

    assert_allclose(
        image1,
        image2,
    )

    likelihood._restore_multiband_offsets(
        backup
    )

    print(
        "PASS: repeated apply gives identical result"
    )

if __name__ == "__main__":
    test_apply_restore_multiband_offsets()
    test_multiband_offsets_change_image_and_restore()
    test_repeated_apply_gives_same_image()
