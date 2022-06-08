import numpy as np

from lenstronomy.Data.psf import PSF
from lenstronomy.Data.kinematic_bin import KinBin

__all__ = ['KinData']

class KinData(object):
    """
    Class which summarizes the binned kinematics and the associated PSF

    """

    def __init__(self,kinematic_bin_class, psf_class):
        """
        :param kinematic_bin_class: KinBin class
        :param psf_class: PSF class

        """
        self.PSF = psf_class
        self.KinBin = kinematic_bin_class