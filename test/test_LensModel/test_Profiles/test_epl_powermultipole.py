__author__ = "eckerl"


import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.LensModel.Profiles.epl_powermultipole import EPL_PMultipol
from lenstronomy.LensModel.Profiles.epl import EPL
from lenstronomy.Util import util

class EPL_PMultipolTest(object):
    """Test analytical kappa vs kappa from Hessian (0.5 * (f_xx + f_yy))."""

    def setup_method(self):
        # Define parameters for the EPL_PMultipol profile
        self.lens_model = EPL_PMultipol() 
        self.epl=EPL() # Correct profile here
        self.theta_E = 1.0
        self.gamma = 2.0
        self.k_m = 0.1
        self.m = 4
        self.phi_m = np.pi / 4
        self.e1 = 0.0  # Example value for eccentricity
        self.e2 = 0.0  # Example value for eccentricity

    def analytical_kappa(self, x, y):
        """Compute analytical kappa."""
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        if r == 0:
            raise ValueError("Kappa is not defined at r = 0")

        kappa = 0.5 * (self.theta_E / r)**(self.gamma - 1) * self.k_m * np.cos(self.m * (phi - self.phi_m))+(3-self.gamma)/2*(self.theta_E/r)**(self.gamma-1)
        return kappa

    def test_kappa_comparison(self):
        """Test analytical kappa vs kappa from Hessian."""
        test_points = [(1, 2), (-0.5, 0.8), (0.5, -0.5), (-1, -1)]
        kwargs = {
            "theta_E": self.theta_E,
            "gamma": self.gamma,
            "e1": self.e1,
            "e2": self.e2,
            "m": self.m,
            "k_m": self.k_m,
            "phi_m": self.phi_m,
        }

        for i, (x_i, y_i) in enumerate(test_points):
            if np.sqrt(x_i**2 + y_i**2) > 0.1:  # Avoid division by zero
                # Compute analytical kappa
                kappa_analytical = self.analytical_kappa(x_i, y_i)

                # Compute numerical f_xx, f_xy, f_yy using EPL_PMultipol
                f_xx, f_xy, _, f_yy = self.lens_model.hessian(x_i, y_i, **kwargs)

                # Compute numerical kappa
                kappa_numerical = 0.5 * (f_xx + f_yy)

                # Compare analytical and numerical kappa
                npt.assert_almost_equal(
                    kappa_analytical, kappa_numerical, decimal=4,
                    err_msg=f"Kappa mismatch at (x, y) = ({x_i}, {y_i})"
                )

                print(
                    f"Point {i+1}: (x, y) = ({x_i}, {y_i}), "
                    f"Relative Difference Kappa = {np.abs(kappa_analytical - kappa_numerical) / np.abs(kappa_analytical):.4e}"
                )
