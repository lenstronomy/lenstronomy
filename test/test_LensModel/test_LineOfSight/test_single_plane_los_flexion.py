__author__ = "nataliehogg"

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.LineOfSight.single_plane_los_flexion import (
    SinglePlaneLOSFlexion,
)
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

    These functions are the same as in TestLensModel but with the addition of LOSF and
    LOSF_MINIMAL as profiles. with all params in self.kwargs_los set to zero, the
    results should be the same as the non-LOSF cases originally tested.
    """

    def setup_method(self):
        self.lensModel = SinglePlane(["GAUSSIAN_POTENTIAL"])
        self.lensModel_los = SinglePlaneLOSFlexion(
            ["GAUSSIAN_POTENTIAL", "LOS_FLEXION"], index_los_flexion=1
        )
        self.lensModel_minimal = SinglePlaneLOSFlexion(
            ["GAUSSIAN_POTENTIAL", "LOS_FLEXION_MINIMAL"], index_los_flexion=1
        )
        self.kwargs = {
            "amp": 1.0,
            "sigma_x": 2.0,
            "sigma_y": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        self.los_kwargs = {
            "kappa_od": 0.0,
            "kappa_os": 0.0,
            "kappa_ds": 0.0,
            "kappa_los": 0.0,
            "gamma1_od": 0.0,
            "gamma2_od": 0.0,
            "gamma1_os": 0.0,
            "gamma2_os": 0.0,
            "gamma1_ds": 0.0,
            "gamma2_ds": 0.0,
            "gamma1_los": 0.0,
            "gamma2_los": 0.0,
            "F1_od": 0.0,
            "F2_od": 0.0,
            "G1_od": 0.0,
            "G2_od": 0.0,
            "F1_os": 0.0,
            "F2_os": 0.0,
            "G1_os": 0.0,
            "G2_os": 0.0,
            "F1_los": 0.0,
            "F2_los": 0.0,
            "G1_los": 0.0,
            "G2_los": 0.0,
            "F1_1ds": 0.0,
            "F2_1ds": 0.0,
            "G1_1ds": 0.0,
            "G2_1ds": 0.0,
            "F1_2ds": 0.0,
            "F2_2ds": 0.0,
            "G1_2ds": 0.0,
            "G2_2ds": 0.0,
            "F1_1los": 0.0,
            "F2_1los": 0.0,
            "G1_1los": 0.0,
            "G2_1los": 0.0,
            "omega_os": 0.0,
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

        lensModel = SinglePlane(["GAUSSIAN"])
        lensModel_los = SinglePlaneLOSFlexion(
            ["GAUSSIAN", "LOS_FLEXION"], index_los_flexion=1
        )
        lensModel_minimal = SinglePlaneLOSFlexion(
            ["GAUSSIAN", "LOS_FLEXION_MINIMAL"], index_los_flexion=1
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

        lensModel = SinglePlane(["GAUSSIAN"])
        lensModel_los = SinglePlaneLOSFlexion(
            ["GAUSSIAN", "LOS_FLEXION"], index_los_flexion=1
        )
        lensModel_minimal = SinglePlaneLOSFlexion(
            ["GAUSSIAN", "LOS_FLEXION_MINIMAL"], index_los_flexion=1
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
        lensModel_los = SinglePlaneLOSFlexion(
            lens_model_list=["SIS", "LOS_FLEXION"], index_los_flexion=1
        )
        density_los = lensModel_los.density(
            r=r, kwargs=[{"theta_E": theta_E}, self.los_kwargs]
        )
        npt.assert_almost_equal(density_los, density_model, decimal=8)

        # LOS_MINIMAL
        lensModel_minimal = SinglePlaneLOSFlexion(
            lens_model_list=["SIS", "LOS_FLEXION_MINIMAL"], index_los_flexion=1
        )
        density_minimal = lensModel_minimal.density(
            r=r, kwargs=[{"theta_E": theta_E}, self.los_kwargs]
        )
        npt.assert_almost_equal(density_minimal, density_model, decimal=8)

    def test_bool_list(self):
        lensModel_los = SinglePlaneLOSFlexion(
            ["SPEP", "SHEAR", "LOS_FLEXION"], index_los_flexion=2
        )
        lensModel_minimal = SinglePlaneLOSFlexion(
            ["SPEP", "SHEAR", "LOS_FLEXION_MINIMAL"], index_los_flexion=2
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
        """This function asserts that the outcome from LOS_FLEXION and LOS_FLEXION
        MINIMAL is the same as MultiPlane.

        Note that the LOS_FLEXION and LOS_FLEXION_MINIMAL models are based on the
        dominant lens approximation, which means that those models are accurate at the
        level of the approximation. The error is of order of the square of the flexion
        parameters magnitude.
        """
        # set up the cosmology to convert between flexions
        # the exact numbers don't matter because we are just doing a comparison

        z_o = 0.0  # redshift of observer
        z_d = 0.5  # redshift of main lens
        z_s = 2.0  # redshift of source

        z_f = (z_o + z_d) / 2  # redshift of foreground perturber
        z_b = (z_d + z_s) / 2  # redshift of background perturber

        # define the flexion parameters of the perturbers. Those are taken in a normal distribution
        # with standard deviation of 0.005 arcsec^-1, in agreement with the expected value of cosmic
        # flexion derived in arXiv:2405.12091.
        F1_f = 0.0007
        F2_f = 0.0016
        G1_f = -0.009
        G2_f = 0.0018
        F1_b = -0.0070
        F2_b = 0.0051
        G1_b = -0.0028
        G2_b = -0.0016

        Flexions_f = np.array([F1_f, F2_f, G1_f, G2_f])
        Flexions_b = np.array([F1_b, F2_b, G1_b, G2_b])

        def d(z1, z2):
            return cosmo.angular_diameter_distance_z1z2(z1, z2).to_value()

        # conversion of the base flexions to LOS flexions using some distance factors
        Flexions_od = Flexions_f * (
            (d(z_o, z_s) * d(z_f, z_d)) / (d(z_o, z_d) * d(z_f, z_s))
        )
        Flexions_os = Flexions_f + Flexions_b
        Flexions_1ds = (
            Flexions_b
            * ((d(z_o, z_s) * d(z_d, z_b)) / (d(z_o, z_b) * d(z_d, z_s))) ** 2
        )
        Flexions_2ds = Flexions_b * (
            (d(z_o, z_s) * d(z_d, z_b)) / (d(z_o, z_b) * d(z_d, z_s))
        )

        # conversion of the base flexions to lenstronomy conventions
        transfer_matrix = np.array(
            [[3, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, 3, 0, -1]]
        )
        Flexions_f_lenstro = np.matmul(transfer_matrix, Flexions_f)
        Flexions_b_lenstro = np.matmul(transfer_matrix, Flexions_b)

        # test three image positions
        x, y = np.array([0.4, 0.2, -1.2]), np.array([-1.7, 1.5, 0])

        # setup the lens models and kwargs

        # Shared in both cases
        kwargs_epl = {
            "theta_E": 1.0,
            "gamma": 1.95,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.07,
            "e2": -0.03,
        }

        # LOSF
        kwargs_los_flexion = {
            "kappa_od": 0.0,
            "kappa_os": 0.0,
            "kappa_ds": 0.0,
            "gamma1_od": 0.0,
            "gamma2_od": 0.0,
            "gamma1_os": 0.0,
            "gamma2_os": 0.0,
            "gamma1_ds": 0.0,
            "gamma2_ds": 0.0,
            "F1_od": Flexions_od[0],
            "F2_od": Flexions_od[1],
            "G1_od": Flexions_od[2],
            "G2_od": Flexions_od[3],
            "F1_os": Flexions_os[0],
            "F2_os": Flexions_os[1],
            "G1_os": Flexions_os[2],
            "G2_os": Flexions_os[3],
            "F1_1ds": Flexions_1ds[0],
            "F2_1ds": Flexions_1ds[1],
            "G1_1ds": Flexions_1ds[2],
            "G2_1ds": Flexions_1ds[3],
            "F1_2ds": Flexions_2ds[0],
            "F2_2ds": Flexions_2ds[1],
            "G1_2ds": Flexions_2ds[2],
            "G2_2ds": Flexions_2ds[3],
            "omega_os": 0.0,
        }

        kwargs_singleplane_los = [kwargs_los_flexion, kwargs_epl]

        lens_model_los = SinglePlaneLOSFlexion(
            ["LOS_FLEXION", "EPL"], index_los_flexion=0
        )

        # Multiplane
        lens_model_list = ["EPL", "FLEXION", "FLEXION"]

        redshift_list = [z_d, z_f, z_b]

        kwargs_flexion_f = {
            "g1": Flexions_f_lenstro[0],
            "g2": Flexions_f_lenstro[1],
            "g3": Flexions_f_lenstro[2],
            "g4": Flexions_f_lenstro[3],
        }
        kwargs_flexion_b = {
            "g1": Flexions_b_lenstro[0],
            "g2": Flexions_b_lenstro[1],
            "g3": Flexions_b_lenstro[2],
            "g4": Flexions_b_lenstro[3],
        }

        kwargs_multiplane = [kwargs_epl, kwargs_flexion_f, kwargs_flexion_b]

        lens_model_multiplane = MultiPlane(
            z_source=z_s,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )

        # set the tolerance
        # In the dominant lens approximation, the neglected terms are of the the order the square of the
        # flexion magnitude times theta^3 (theta ~ 1 arcsec). We therefore set the absolute tolerance to
        # 10 x 0.005^2 = 2.5e-4 arcsec.
        atolerance = 2.5e-4
        # For time delay though absolute tolerance is not relevant because terms are multiplied to the
        # time delay scale; the relative tolerance is therefore more suitable.
        rtolerance = 1e-3

        # compare some different results from single_plane_los and multiplane
        # we use assert_allclose rather than assert_almost_equal because we are dealing with arrays
        # since we pass an array of image positions

        # displacement angle
        alpha_multiplane_x, alpha_multiplane_y = lens_model_multiplane.alpha(
            x, y, kwargs_multiplane
        )
        alpha_los_x, alpha_los_y = lens_model_los.alpha(x, y, kwargs_singleplane_los)

        npt.assert_allclose(alpha_multiplane_x, alpha_los_x, atol=atolerance)
        npt.assert_allclose(alpha_multiplane_y, alpha_los_y, atol=atolerance)

        # ray_shooting
        beta_multiplane_x, beta_multiplane_y = lens_model_multiplane.ray_shooting(
            x, y, kwargs_multiplane
        )
        beta_los_x, beta_los_y = lens_model_los.ray_shooting(
            x, y, kwargs_singleplane_los
        )

        npt.assert_allclose(beta_multiplane_x, beta_los_x, atol=atolerance)
        npt.assert_allclose(beta_multiplane_y, beta_los_y, atol=atolerance)

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
        npt.assert_allclose(hessian_multiplane_xx, hessian_los_xx, atol=atolerance)
        npt.assert_allclose(hessian_multiplane_xy, hessian_los_xy, atol=atolerance)
        npt.assert_allclose(hessian_multiplane_yx, hessian_los_yx, atol=atolerance)
        npt.assert_allclose(hessian_multiplane_yy, hessian_los_yy, atol=atolerance)

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

        lens_model_los_time = LensModel(
            ["LOS_FLEXION", "EPL"], z_lens=z_d, z_source=z_s
        )
        kwargs_time_los = [kwargs_los_flexion, kwargs_epl]

        los_solver = LensEquationSolver(lens_model_los_time)
        x_image_los, y_image_los = los_solver.findBrightImage(
            ra_source, dec_source, kwargs_time_los, numImages=number_of_images
        )

        t_days_los = lens_model_los_time.arrival_time(
            x_image_los, y_image_los, kwargs_time_los
        )
        dt_days_los = t_days_los[1:] - t_days_los[0]

        npt.assert_allclose(dt_days_mp, dt_days_los, rtol=rtolerance)

    def test_init(self):
        lens_model_list = [
            "TNFW",
            "TRIPLE_CHAMELEON",
            "LOS_FLEXION",
            "SHEAR_GAMMA_PSI",
            "CURVED_ARC_CONST",
            "NFW_MC",
            "ARC_PERT",
            "MULTIPOLE",
            "MULTIPOLE_ELL",
            "CURVED_ARC_SPP",
        ]
        lensModel = SinglePlaneLOSFlexion(
            lens_model_list=lens_model_list, index_los_flexion=2
        )
        assert lensModel.func_list[0].param_names[0] == "Rs"


class TestRaise(unittest.TestCase):
    def test_raise(self):
        """Check whether raises occurs if fastell4py is not installed.

        :return:
        """
        if bool_test is False:
            with self.assertRaises(ImportError):
                SinglePlaneLOSFlexion(
                    lens_model_list=["PEMD", "LOS_FLEXION"], index_los_flexion=1
                )
            with self.assertRaises(ImportError):
                SinglePlaneLOSFlexion(
                    lens_model_list=["SPEMD", "LOS_FLEXION"], index_los_flexion=1
                )
        else:
            SinglePlaneLOSFlexion(
                lens_model_list=["PEMD", "SPEMD", "LOS_FLEXION"], index_los_flexion=2
            )


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
