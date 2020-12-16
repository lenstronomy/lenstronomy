__author__ = "lynevdv"

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['ElliSLICE']

class ElliSLICE (LensProfileBase):
    """
    This class computes the lensing quantities for an elliptical slice of constant density.
    Based on Schramm 1994 https://ui.adsabs.harvard.edu/abs/1994A%26A...284...44S/abstract

    Computes the lensing quantities of an elliptical slice with semi major axis 'a' and
    semi minor axis 'b', centered on 'center_x' and 'center_y', oriented with an angle 'psi'
    in radian, and with constant surface mass density 'sigma_0'

    """