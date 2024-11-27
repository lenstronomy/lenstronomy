__author__ = "nataliehogg"

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.LineOfSightFlexion.single_plane_los_flexion import SinglePlaneLOSFlexion
from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane
from lenstronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from astropy.cosmology import default_cosmology

cosmo = default_cosmology.get()

try:
    import fastell4py

    bool_test = True
except:
    bool_test = False


class TestSinglePlaneLOSFlexion(object):
    """Tests the SinglePlaneLOSFlexion routines.

    these functions are the same as in TestLensModel but with the addition of LOSF and
    LOSF_MINIMAL as profiles. with all params in self.kwargs_los set to zero, the results
    should be the same as the non-LOSF cases originally tested. 
    """

    def setup_method(self):
        self.lensModel = SinglePlane(["GAUSSIAN_POTENTIAL"])
        self.lensModel_los = SinglePlaneLOSFlexion(["GAUSSIAN_POTENTIAL", "LOSF"], index_losf=1)
        self.lensModel_minimal = SinglePlaneLOSFlexion(
            ["GAUSSIAN_POTENTIAL", "LOSF_MINIMAL"], index_losf=1
        )
        self.kwargs = {
            "amp": 1.0,
            "sigma_x": 2.0,
            "sigma_y": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.los_kwargs = {
            'kappa_od': 0.0, 
            'kappa_os': 0.0, 
            'kappa_ds': 0.0,
            'kappa_los': 0.0,

            'gamma1_od': 0.0, 
            'gamma2_od': 0.0,
            'gamma1_os': 0.0,
            'gamma2_os': 0.0,
            'gamma1_ds': 0.0, 
            'gamma2_ds': 0.0, 
            'gamma1_los': 0.0,
            'gamma2_los': 0.0,

            'F1_od': 0.0, 
            'F2_od': 0.0, 
            'G1_od': 0.0, 
            'G2_od': 0.0,

            'F1_os': 0.0, 
            'F2_os': 0.0, 
            'G1_os': 0.0, 
            'G2_os': 0.0, 

            'F1_los': 0.0,
            'F2_los': 0.0,
            'G1_los': 0.0,
            'G2_los': 0.0,

            'F1_1ds': 0.0, 
            'F2_1ds': 0.0,
            'G1_1ds': 0.0, 
            'G2_1ds': 0.0, 

            'F1_2ds': 0.0, 
            'F2_2ds': 0.0, 
            'G1_2ds': 0.0, 
            'G2_2ds': 0.0, 

            'F1_1los': 0.0, 
            'F2_1los': 0.0, 
            'G1_1los': 0.0, 
            'G2_1los': 0.0,

            'omega_os': 0.0,
            'omega_los': 0.0,
            }

            

    def test_potential(self):
        output = self.lensModel.potential(x=1.0, y=1.0, kwargs=[self.kwargs])
        output_los = self.lensModel_los.potential(
            x=1.0, y=1.0, kwargs=[self.kwargs, self.los_kwargs]
        )
        output_minimal = self.lensModel_minimal.potential(
            x=1.0, y=1.0, kwargs=[self.kwargs, self.los_kwargs]
        )
        npt.assert_almost_equal(output_los, output, decimal=8)
        npt.assert_almost_equal(output_minimal, output, decimal=8)

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(x=1.0, y=1.0, kwargs=[self.kwargs])
        output1_los, output2_los = self.lensModel_los.alpha(
            x=1.0, y=1.0, kwargs=[self.kwargs, self.los_kwargs]
        )
        output1_minimal, output2_minimal = self.lensModel_minimal.alpha(
            x=1.0, y=1.0, kwargs=[self.kwargs, self.los_kwargs]
        )
        npt.assert_almost_equal(output1_los, output1, decimal=8)
        npt.assert_almost_equal(output2_los, output2, decimal=8)
        npt.assert_almost_equal(output1_minimal, output1, decimal=8)
        npt.assert_almost_equal(output2_minimal, output2, decimal=8)

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(
            x=1.0, y=1.0, kwargs=[self.kwargs]
        )
        delta_x_los, delta_y_los = self.lensModel_los.ray_shooting(
            x=1.0, y=1.0, kwargs=[self.kwargs, self.los_kwargs]
        )
        delta_x_minimal, delta_y_minimal = self.lensModel_minimal.ray_shooting(
            x=1.0, y=1.0, kwargs=[self.kwargs, self.los_kwargs]
        )
        npt.assert_almost_equal(delta_x_los, delta_x, decimal=8)
        npt.assert_almost_equal(delta_y_los, delta_y, decimal=8)
        npt.assert_almost_equal(delta_x_minimal, delta_x, decimal=8)
        npt.assert_almost_equal(delta_y_minimal, delta_y, decimal=8)

    def test_mass_2d(self):
        mass_kwargs = {"amp": 1.0, "sigma": 2.0, "center_x": 0.0, "center_y": 0.0}

        lensModel = SinglePlane(["GAUSSIAN"])
        lensModel_los = SinglePlaneLOSFlexion(["GAUSSIAN", "LOSF"], index_losf=1)
        lensModel_minimal = SinglePlaneLOSFlexion(["GAUSSIAN", "LOSF_MINIMAL"], index_losf=1)

        output = lensModel.mass_2d(r=1, kwargs=[mass_kwargs])
        output_los = lensModel_los.mass_2d(r=1, kwargs=[mass_kwargs, self.los_kwargs])
        output_minimal = lensModel_minimal.mass_2d(
            r=1, kwargs=[mass_kwargs, self.los_kwargs]
        )

        npt.assert_almost_equal(output_los, output, decimal=8)
        npt.assert_almost_equal(output_minimal, output, decimal=8)

    def test_mass_3d(self):
        mass_kwargs = {"amp": 1.0, "sigma": 2.0, "center_x": 0.0, "center_y": 0.0}

        lensModel = SinglePlane(["GAUSSIAN"])
        lensModel_los = SinglePlaneLOSFlexion(["GAUSSIAN", "LOSF"], index_losf=1)
        lensModel_minimal = SinglePlaneLOSFlexion(["GAUSSIAN", "LOSF_MINIMAL"], index_losf=1)

        output = lensModel.mass_3d(r=1, kwargs=[mass_kwargs])
        output_los = lensModel_los.mass_3d(r=1, kwargs=[mass_kwargs, self.los_kwargs])
        output_minimal = lensModel_minimal.mass_3d(
            r=1, kwargs=[mass_kwargs, self.los_kwargs]
        )

        npt.assert_almost_equal(output_los, output, decimal=8)
        npt.assert_almost_equal(output_minimal, output, decimal=8)

    def test_density(self):
        theta_E = 1
        r = 1
        sis = SIS()
        density_model = sis.density_lens(r=r, theta_E=theta_E)

        # LOS
        lensModel_los = SinglePlaneLOSFlexion(lens_model_list=["SIS", "LOSF"], index_losf=1)
        density_los = lensModel_los.density(
            r=r, kwargs=[{"theta_E": theta_E}, self.los_kwargs]
        )
        npt.assert_almost_equal(density_los, density_model, decimal=8)

        # LOS_MINIMAL
        lensModel_minimal = SinglePlaneLOSFlexion(
            lens_model_list=["SIS", "LOSF_MINIMAL"], index_losf=1
        )
        density_minimal = lensModel_minimal.density(
            r=r, kwargs=[{"theta_E": theta_E}, self.los_kwargs]
        )
        npt.assert_almost_equal(density_minimal, density_model, decimal=8)

    def test_bool_list(self):
        lensModel_los = SinglePlaneLOSFlexion(["SPEP", "SHEAR", "LOSF"], index_losf=2)
        lensModel_minimal = SinglePlaneLOSFlexion(
            ["SPEP", "SHEAR", "LOSF_MINIMAL"], index_losf=2
        )
        kwargs = [
            {
                "theta_E": 1,
                "gamma": 2,
                "e1": 0.1,
                "e2": -0.1,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.01, "gamma2": -0.02},
            self.los_kwargs,
        ]

        # LOS
        alphax_1_los, alphay_1_los = lensModel_los.alpha(1, 1, kwargs, k=0)
        alphax_1_list, alphay_1_list = lensModel_los.alpha(1, 1, kwargs, k=[0])
        npt.assert_almost_equal(alphax_1_los, alphax_1_list, decimal=5)
        npt.assert_almost_equal(alphay_1_los, alphay_1_list, decimal=5)

        alphax_1_1_los, alphay_1_1_los = lensModel_los.alpha(1, 1, kwargs, k=0)
        alphax_1_2_los, alphay_1_2_los = lensModel_los.alpha(1, 1, kwargs, k=1)
        alphax_full, alphay_full = lensModel_los.alpha(1, 1, kwargs, k=None)
        npt.assert_almost_equal(alphax_1_1_los + alphax_1_2_los, alphax_full, decimal=5)
        npt.assert_almost_equal(alphay_1_1_los + alphay_1_2_los, alphay_full, decimal=5)

        # LOS_MINIMAL
        alphax_1_minimal, alphay_1_minimal = lensModel_minimal.alpha(1, 1, kwargs, k=0)
        alphax_1_list_minimal, alphay_1_list_minimal = lensModel_minimal.alpha(
            1, 1, kwargs, k=[0]
        )
        npt.assert_almost_equal(alphax_1_minimal, alphax_1_list_minimal, decimal=5)
        npt.assert_almost_equal(alphay_1_minimal, alphay_1_list_minimal, decimal=5)

        alphax_1_1_minimal, alphay_1_1_minimal = lensModel_minimal.alpha(
            1, 1, kwargs, k=0
        )
        alphax_1_2_minimal, alphay_1_2_minimal = lensModel_minimal.alpha(
            1, 1, kwargs, k=1
        )
        alphax_full_minimal, alphay_full_minimal = lensModel_minimal.alpha(
            1, 1, kwargs, k=None
        )
        npt.assert_almost_equal(
            alphax_1_1_minimal + alphax_1_2_minimal, alphax_full_minimal, decimal=5
        )
        npt.assert_almost_equal(
            alphay_1_1_minimal + alphay_1_2_minimal, alphay_full_minimal, decimal=5
        )

    def test_init(self):
        lens_model_list = [
            "TNFW",
            "TRIPLE_CHAMELEON",
            "LOSF",
            "SHEAR_GAMMA_PSI",
            "CURVED_ARC_CONST",
            "NFW_MC",
            "ARC_PERT",
            "MULTIPOLE",
            "MULTIPOLE_ELL",
            "CURVED_ARC_SPP",
        ]
        lensModel = SinglePlaneLOSFlexion(lens_model_list=lens_model_list, index_losf=0)
        assert lensModel.func_list[0].param_names[0] == "Rs"


class TestRaise(unittest.TestCase):
    def test_raise(self):
        """Check whether raises occurs if fastell4py is not installed.

        :return:
        """
        if bool_test is False:
            with self.assertRaises(ImportError):
                SinglePlaneLOSFlexion(lens_model_list=["PEMD", "LOSF"], index_losf=1)
            with self.assertRaises(ImportError):
                SinglePlaneLOSFlexion(lens_model_list=["SPEMD", "LOSF"], index_losf=1)
        else:
            SinglePlaneLOSFlexion(lens_model_list=["PEMD", "SPEMD", "LOSF"], index_losf=2)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
