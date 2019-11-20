__author__ = 'aymgal'

# implementations of proximal operators adapted to sparsity

import numpy as np
from lenstronomy.Util import util


def prox_sparsity_wavelets(coeffs, step, level_const=None, level_pixels=None,
                           norm=1, force_positivity=False):

    n_scales = coeffs.shape[0]

    if level_const is None:
        level_const = np.ones(n_scales)

    if level_pixels is None:
        level_pixels = np.ones_like(coeffs)

    # apply threshold operation to all starlet scales except the coarsest
    for l in range(n_scales-1):
        level_eff = step * level_const[l] * level_pixels[l, :, :]
        if norm == 0:
            coeffs[l, :, :] = util.hard_threshold(coeffs[l, :, :], level_eff)
        else:
            coeffs[l, :, :] = util.soft_threshold(coeffs[l, :, :], level_eff)

    return coeffs


def prox_positivity(image):
    image[image < 0] = 0.
    return image
