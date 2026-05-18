__author__ = "nan zhang"

import numpy.testing as npt
import numpy as np

from lenstronomy.Util import primary_beam_util


def test_primary_beam_value_at_coords():

    numPix = 100
    primary_beam = np.zeros((numPix, numPix))
    for i in range(numPix):
        for j in range(numPix):
            primary_beam[i, j] = np.exp(-1e-4 * ((i - 78) ** 2 + (j - 56) ** 2))
    primary_beam /= np.max(primary_beam)

    # Test scalar numbers as input
    output1 = primary_beam_util.primary_beam_value_at_coords(0, 1.2, primary_beam)
    npt.assert_almost_equal(output1, 0.405407, decimal=8)

    # Test points outside of the image sent to zero
    output2_1 = primary_beam_util.primary_beam_value_at_coords(-0.001, 10, primary_beam)
    output2_2 = primary_beam_util.primary_beam_value_at_coords(
        numPix - 1 + 0.001, 10, primary_beam
    )
    output2_3 = primary_beam_util.primary_beam_value_at_coords(50, -0.001, primary_beam)
    output2_4 = primary_beam_util.primary_beam_value_at_coords(
        50, numPix - 1 + 0.001, primary_beam
    )
    assert output2_1 == 0.0
    assert output2_2 == 0.0
    assert output2_3 == 0.0
    assert output2_4 == 0.0

    # Test array as input
    x_pos = np.array([1.2, 89, -2, 6, 43.7])
    y_pos = np.array([5.73, 15.2, 10.6, 102, 43])
    output3 = primary_beam_util.primary_beam_value_at_coords(x_pos, y_pos, primary_beam)
    npt.assert_almost_equal(
        output3, np.array([0.4394668, 0.60454208, 0.0, 0.0, 0.87142193]), decimal=8
    )
