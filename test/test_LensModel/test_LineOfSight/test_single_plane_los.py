__author__ = "nataliehogg"

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.LineOfSight.single_plane_los import SinglePlaneLOS
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


class TestSinglePlaneLOS(object):
    """Tests the SinglePlaneLOS routines.

    these functions are the same as in TestLensModel but with the addition of LOS and
    LOS_MINIMAL as profiles. with all params in self.kwargs_los set to zero, the results
    should be the same as the non-LOS cases originally tested the test_los_vs_multiplane
    checks that a multiplane setup with three shear planes returns the same as the LOS
    and LOS MINIMAL models
    """

    def setup_method(self):
        self.lensModel = SinglePlane(["GAUSSIAN"])
        self.lensModel_los = SinglePlaneLOS(["GAUSSIAN", "LOS"], index_los=1)
        self.lensModel_minimal = SinglePlaneLOS(
            ["GAUSSIAN", "LOS_MINIMAL"], index_los=1
        )
        self.kwargs = {
            "amp": 1.0,
            "sigma_x": 2.0,
            "sigma_y": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.los_kwargs = {
            "gamma1_os": 0.0,
            "gamma2_os": 0.0,
            "kappa_os": 0.0,
            "omega_os": 0.0,
            "gamma1_od": 0.0,
            "gamma2_od": 0.0,
            "kappa_od": 0.0,
            "omega_od": 0.0,
            "gamma1_ds": 0.0,
            "gamma2_ds": 0.0,
            "kappa_ds": 0.0,
            "omega_ds": 0.0,
            "gamma1_los": 0.0,
            "gamma2_los": 0.0,
            "kappa_los": 0.0,
            "omega_los": 0.0,
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

        lensModel = SinglePlane(["GAUSSIAN_KAPPA"])
        lensModel_los = SinglePlaneLOS(["GAUSSIAN_KAPPA", "LOS"], index_los=1)
        lensModel_minimal = SinglePlaneLOS(
            ["GAUSSIAN_KAPPA", "LOS_MINIMAL"], index_los=1
        )

        output = lensModel.mass_2d(r=1, kwargs=[mass_kwargs])
        output_los = lensModel_los.mass_2d(r=1, kwargs=[mass_kwargs, self.los_kwargs])
        output_minimal = lensModel_minimal.mass_2d(
            r=1, kwargs=[mass_kwargs, self.los_kwargs]
        )

        npt.assert_almost_equal(output_los, output, decimal=8)
        npt.assert_almost_equal(output_minimal, output, decimal=8)

    def test_mass_3d(self):
        mass_kwargs = {"amp": 1.0, "sigma": 2.0, "center_x": 0.0, "center_y": 0.0}

        lensModel = SinglePlane(["GAUSSIAN_KAPPA"])
        lensModel_los = SinglePlaneLOS(["GAUSSIAN_KAPPA", "LOS"], index_los=1)
        lensModel_minimal = SinglePlaneLOS(
            ["GAUSSIAN_KAPPA", "LOS_MINIMAL"], index_los=1
        )

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
        lensModel_los = SinglePlaneLOS(lens_model_list=["SIS", "LOS"], index_los=1)
        density_los = lensModel_los.density(
            r=r, kwargs=[{"theta_E": theta_E}, self.los_kwargs]
        )
        npt.assert_almost_equal(density_los, density_model, decimal=8)

        # LOS_MINIMAL
        lensModel_minimal = SinglePlaneLOS(
            lens_model_list=["SIS", "LOS_MINIMAL"], index_los=1
        )
        density_minimal = lensModel_minimal.density(
            r=r, kwargs=[{"theta_E": theta_E}, self.los_kwargs]
        )
        npt.assert_almost_equal(density_minimal, density_model, decimal=8)

    def test_bool_list(self):
        lensModel_los = SinglePlaneLOS(["SPEP", "SHEAR", "LOS"], index_los=2)
        lensModel_minimal = SinglePlaneLOS(
            ["SPEP", "SHEAR", "LOS_MINIMAL"], index_los=2
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

    def test_los_versus_multiplane(self):
        """This function asserts that the outcome from LOS and LOS MINIMAL is the same
        as MultiPlane."""
        # set up the cosmology to convert between shears
        # the exact numbers don't matter because we are just doing a comparison

        z_o = 0.0  # redshift of observer
        z_d = 0.5  # redshift of main lens
        z_s = 2.0  # redshift of source

        z_f = (z_o + z_d) / 2
        z_b = (z_d + z_s) / 2

        gamma1_od = 0.05
        gamma2_od = -0.01
        gamma1_os = 0.02
        gamma2_os = 0.1
        gamma1_ds = -0.04
        gamma2_ds = 0.03

        def d(z1, z2):
            return cosmo.angular_diameter_distance_z1z2(z1, z2).to_value()

        # conversion of linear LOS shears to lenstronomy convention
        gamma1_f = gamma1_od * (
            (d(z_o, z_d) * d(z_f, z_s)) / (d(z_o, z_s) * d(z_f, z_d))
        )
        gamma2_f = gamma2_od * (
            (d(z_o, z_d) * d(z_f, z_s)) / (d(z_o, z_s) * d(z_f, z_d))
        )

        gamma1_b = gamma1_ds * (
            (d(z_o, z_b) * d(z_d, z_s)) / (d(z_o, z_s) * d(z_d, z_b))
        )
        gamma2_b = gamma2_ds * (
            (d(z_o, z_b) * d(z_d, z_s)) / (d(z_o, z_s) * d(z_d, z_b))
        )

        gamma1_d = gamma1_os - gamma1_f - gamma1_b
        gamma2_d = gamma2_os - gamma2_f - gamma2_b

        # compute non-linear correction to os term
        Identity = np.identity(2)

        Gamma_f = np.array([[gamma1_f, gamma2_f], [gamma2_f, -gamma1_f]])

        Gamma_d = np.array([[gamma1_d, gamma2_d], [gamma2_d, -gamma1_d]])

        Gamma_b = np.array([[gamma1_b, gamma2_b], [gamma2_b, -gamma1_b]])

        Gamma_od = np.array([[gamma1_od, gamma2_od], [gamma2_od, -gamma1_od]])

        Gamma_ofb = np.array(Gamma_f) * (
            (d(z_o, z_s) * d(z_f, z_b)) / (d(z_o, z_b) * d(z_f, z_s))
        )

        Gamma_odb = np.array(Gamma_d) * (
            (d(z_o, z_s) * d(z_d, z_b)) / (d(z_o, z_b) * d(z_d, z_s))
        )

        Gamma_os = (
            Gamma_f
            + Gamma_d
            + Gamma_b
            - np.matmul(Gamma_d, Gamma_od)
            - np.matmul(Gamma_b, Gamma_ofb + np.matmul(Gamma_odb, Identity - Gamma_od))
        )

        kappa_os = (Gamma_os[0, 0] + Gamma_os[1, 1]) / 2
        omega_os = (Gamma_os[1, 0] - Gamma_os[0, 1]) / 2
        gamma1_os = (Gamma_os[0, 0] - Gamma_os[1, 1]) / 2
        gamma2_os = (Gamma_os[0, 1] + Gamma_os[1, 0]) / 2

        # test three image positions
        x, y = np.array([3, 4, 5]), np.array([2, 1, 0])

        lens_model_list = ["EPL", "SHEAR", "SHEAR", "SHEAR"]

        redshift_list = [z_d, z_f, z_d, z_b]

        kwargs_los = {
            "kappa_os": kappa_os,
            "omega_os": omega_os,
            "gamma1_os": gamma1_os,
            "gamma2_os": gamma2_os,
            "kappa_od": 0.0,
            "omega_od": 0.0,
            "gamma1_od": gamma1_od,
            "gamma2_od": gamma2_od,
            "kappa_ds": 0.0,
            "omega_ds": 0.0,
            "gamma1_ds": gamma1_ds,
            "gamma2_ds": gamma2_ds,
        }

        kwargs_epl = {
            "theta_E": 0.8,
            "gamma": 1.95,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.07,
            "e2": -0.03,
        }

        kwargs_gamma_f = {"gamma1": gamma1_f, "gamma2": gamma2_f}
        kwargs_gamma_d = {"gamma1": gamma1_d, "gamma2": gamma2_d}
        kwargs_gamma_b = {"gamma1": gamma1_b, "gamma2": gamma2_b}

        kwargs_singleplane_los = [kwargs_los, kwargs_epl]

        lens_model_los = SinglePlaneLOS(["LOS", "EPL"], index_los=0)

        kwargs_multiplane = [kwargs_epl, kwargs_gamma_f, kwargs_gamma_d, kwargs_gamma_b]

        lens_model_multiplane = MultiPlane(
            z_source=z_s,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        # set the tolerance
        # ray shooting passes at 1e-16
        # hessian around 1e-6
        tolerance = 1e-5

        # compare some different results from single_plane_los and multiplane
        # we use assert_allclose rather than assert_almost_equal because we are dealing with arrays
        # since we pass an array of image positions

        # displacement angle
        alpha_multiplane_x, alpha_multiplane_y = lens_model_multiplane.alpha(
            x, y, kwargs_multiplane
        )
        alpha_los_x, alpha_los_y = lens_model_los.alpha(x, y, kwargs_singleplane_los)
        npt.assert_allclose(alpha_multiplane_x, alpha_los_x, rtol=tolerance)
        npt.assert_allclose(alpha_multiplane_y, alpha_los_y, rtol=tolerance)

        # ray_shooting
        beta_multiplane_x, beta_multiplane_y = lens_model_multiplane.ray_shooting(
            x, y, kwargs_multiplane
        )
        beta_los_x, beta_los_y = lens_model_los.ray_shooting(
            x, y, kwargs_singleplane_los
        )
        npt.assert_allclose(beta_multiplane_x, beta_los_x, rtol=tolerance)
        npt.assert_allclose(beta_multiplane_y, beta_los_y, rtol=tolerance)

        # hessian
        (
            hessian_multiplane_xx,
            hessian_multiplane_xy,
            hessian_multiplane_yx,
            hessian_multiplane_yy,
        ) = lens_model_multiplane.hessian(x, y, kwargs_multiplane)
        (
            hessian_los_xx,
            hessian_los_xy,
            hessian_los_yx,
            hessian_los_yy,
        ) = lens_model_los.hessian(x, y, kwargs_singleplane_los)
        npt.assert_allclose(hessian_multiplane_xx, hessian_los_xx, rtol=tolerance)
        npt.assert_allclose(hessian_multiplane_xy, hessian_los_xy, rtol=tolerance)
        npt.assert_allclose(hessian_multiplane_yx, hessian_los_yx, rtol=tolerance)
        npt.assert_allclose(hessian_multiplane_yy, hessian_los_yy, rtol=tolerance)

        # time delays
        ra_source, dec_source = 0.05, 0.02
        number_of_images = 4

        lens_model_multiplane_time = LensModel(
            lens_model_list,
            z_lens=z_d,
            z_source=z_s,
            lens_redshift_list=redshift_list,
            multi_plane=True,
        )

        multiplane_solver = LensEquationSolver(lens_model_multiplane_time)
        x_image_mp, y_image_mp = multiplane_solver.findBrightImage(
            ra_source, dec_source, kwargs_multiplane, numImages=number_of_images
        )

        t_days_mp = lens_model_multiplane_time.arrival_time(
            x_image_mp, y_image_mp, kwargs_multiplane
        )
        dt_days_mp = t_days_mp[1:] - t_days_mp[0]

        lens_model_los_time = LensModel(["LOS", "EPL"], z_lens=z_d, z_source=z_s)
        kwargs_time_los = [kwargs_los, kwargs_epl]

        los_solver = LensEquationSolver(lens_model_los_time)
        x_image_los, y_image_los = los_solver.findBrightImage(
            ra_source, dec_source, kwargs_time_los, numImages=number_of_images
        )

        t_days_los = lens_model_los_time.arrival_time(
            x_image_los, y_image_los, kwargs_time_los
        )
        dt_days_los = t_days_los[1:] - t_days_los[0]

        npt.assert_allclose(dt_days_mp, dt_days_los, rtol=tolerance)

    def test_init(self):
        # need to do this for los minimal too?
        lens_model_list = [
            "TNFW",
            "TRIPLE_CHAMELEON",
            "LOS",
            "SHEAR_GAMMA_PSI",
            "CURVED_ARC_CONST",
            "NFW_MC",
            "ARC_PERT",
            "MULTIPOLE",
            "CURVED_ARC_SPP",
        ]
        lensModel = SinglePlaneLOS(lens_model_list=lens_model_list, index_los=0)
        assert lensModel.func_list[0].param_names[0] == "Rs"


class TestRaise(unittest.TestCase):
    def test_raise(self):
        """Check whether raises occurs if fastell4py is not installed.

        :return:
        """
        if bool_test is False:
            with self.assertRaises(ImportError):
                SinglePlaneLOS(lens_model_list=["PEMD", "LOS"], index_los=1)
            with self.assertRaises(ImportError):
                SinglePlaneLOS(lens_model_list=["SPEMD", "LOS"], index_los=1)
        else:
            SinglePlaneLOS(lens_model_list=["PEMD", "SPEMD", "LOS"], index_los=2)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
