__author__ = "nan zhang"

import numpy as np
from scipy.ndimage import map_coordinates

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
def primary_beam_value_at_coords(x_pos, y_pos, primary_beam, order=3):
    """Interpolate the primary beam values at specified pixel coordinates. The
    coordinates falling outside the image are assigned to constant zero.

    :param x_pos: array or scalar of x-pixel-coordinates.
    :param y_pos: array or scalar of y-pixel-coordinates.
    :param primary_beam: the primary_beam map
    :param order: the order of the spline interpolation
    :return: a numpy array of the interpolated primary beam values
    """

    primary_beam_interpolated_values = map_coordinates(
        input=primary_beam,
        coordinates=np.vstack([y_pos, x_pos]),
        order=order,
        mode="constant",
        cval=0,
    )

    return primary_beam_interpolated_values
