from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import param_util
import numpy as np


class ArcPerturbations(LensProfileBase):
    """
    uses radial and tangential fourier modes within a specific range in both directions to perturb a lensing potential
    """
    def __init__(self):
        super(ArcPerturbations, self).__init__()

    def function(self, x, y, coeffs, center_x, center_y):
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)