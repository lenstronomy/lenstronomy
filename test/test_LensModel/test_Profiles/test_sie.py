__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LensModel.Profiles.spemd import SPEMD

import numpy as np
import pytest


class TestSIS(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sie = SIE()
        self.spemd = SPEMD()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.
        q = 0.9
        phi_G = 1.
        values = self.sie.function(x, y, theta_E, q, phi_G)
        gamma = 2
        values_spemd = self.spemd.function(x, y, theta_E, gamma, q, phi_G)
        assert values == values_spemd

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.
        q = 0.9
        phi_G = 1.
        values = self.sie.derivatives(x, y, theta_E, q, phi_G)
        gamma = 2
        values_spemd = self.spemd.derivatives(x, y, theta_E, gamma, q, phi_G)
        assert values == values_spemd

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.
        q = 0.9
        phi_G = 1.
        values = self.sie.hessian(x, y, theta_E, q, phi_G)
        gamma = 2
        values_spemd = self.spemd.hessian(x, y, theta_E, gamma, q, phi_G)
        assert values[0] == values_spemd[0]


if __name__ == '__main__':
    pytest.main()