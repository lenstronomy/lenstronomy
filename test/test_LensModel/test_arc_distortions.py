__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.curved_arc_spp import CurvedArcSPP
from lenstronomy.LensModel.Profiles.curved_arc_tan_diff import CurvedArcTanDiff
from lenstronomy.Util import util
from lenstronomy.Util import param_util


class TestArcDistortions(object):
    """Tests the source model routines."""

    def setup_method(self):
        pass

    def test_radial_tangential_distortions(self):
        lens_model_list = ["CURVED_ARC_SPP", "SHEAR", "FLEXION"]
        center_x, center_y = 0.01, 0
        curvature = 1.0 / 2
        lens = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [
            {
                "tangential_stretch": 10,
                "radial_stretch": 1.0,
                "curvature": curvature,
                "direction": -10,
                "center_x": center_x,
                "center_y": center_y,
            },
            {"gamma1": -0.0, "gamma2": -0.0},
            {"g1": 0.0, "g2": 0.0, "g3": -0.0, "g4": 0},
        ]

        extensions = LensModelExtensions(lensModel=lens)

        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = extensions.radial_tangential_differentials(
            x=center_x, y=center_y, kwargs_lens=kwargs_lens, smoothing_3rd=0.0001
        )
        print(orientation_angle, "orientation angle")
        l = 1.0 / dphi_tan_dtan
        npt.assert_almost_equal(l, 1.0 / curvature, decimal=6)

    def test_hessian_eigenvector_mp(self):
        lens_model_list = ["SIS", "SHEAR"]
        lens_mp = LensModel(
            lens_model_list=lens_model_list,
            lens_redshift_list=[0.5, 0.4],
            multi_plane=True,
            z_source=2,
        )
        lens = LensModel(lens_model_list=lens_model_list)
        x0, y0 = 1.0, 1.0
        kwargs_lens = [
            {"theta_E": 1, "center_x": 0, "center_y": 0},
            {"gamma1": 0.0, "gamma2": 0.00001},
        ]

        extensions = LensModelExtensions(lensModel=lens)
        extensions_mp = LensModelExtensions(lensModel=lens_mp)
        w0, w1, v11, v12, v21, v22 = extensions.hessian_eigenvectors(
            x0, y0, kwargs_lens, diff=None
        )
        (
            w0_mp,
            w1_mp,
            v11_mp,
            v12_mp,
            v21_mp,
            v22_mp,
        ) = extensions_mp.hessian_eigenvectors(x0, y0, kwargs_lens, diff=None)
        npt.assert_almost_equal(w0, w0_mp, decimal=3)
        npt.assert_almost_equal(w1, w1_mp, decimal=3)
        npt.assert_almost_equal(v11, v11_mp, decimal=3)
        npt.assert_almost_equal(v12, v12_mp, decimal=3)
        npt.assert_almost_equal(v21, v21_mp, decimal=3)
        npt.assert_almost_equal(v22, v22_mp, decimal=3)

    def test_radial_tangential_stretch(self):
        lens_model_list = ["SIS", "SHEAR"]
        lens_mp = LensModel(
            lens_model_list=lens_model_list,
            lens_redshift_list=[0.5, 0.4],
            multi_plane=True,
            z_source=2,
        )
        lens = LensModel(lens_model_list=lens_model_list)
        x0, y0 = 1.0, 1.0

        kwargs_lens = [
            {"theta_E": 1, "center_x": 0, "center_y": 0},
            {"gamma1": 0.0, "gamma2": 0.00001},
        ]

        extensions = LensModelExtensions(lensModel=lens)
        extensions_mp = LensModelExtensions(lensModel=lens_mp)

        (
            radial_stretch,
            tangential_stretch,
            v_rad1,
            v_rad2,
            v_tang1,
            v_tang2,
        ) = extensions.radial_tangential_stretch(x0, y0, kwargs_lens, diff=None)
        (
            radial_stretch_mp,
            tangential_stretch_mp,
            v_rad1_mp,
            v_rad2_mp,
            v_tang1_mp,
            v_tang2_mp,
        ) = extensions_mp.radial_tangential_stretch(x0, y0, kwargs_lens, diff=None)
        npt.assert_almost_equal(radial_stretch, radial_stretch_mp, decimal=4)
        npt.assert_almost_equal(tangential_stretch, tangential_stretch_mp, decimal=4)
        npt.assert_almost_equal(v_rad1, v_rad1_mp, decimal=4)
        npt.assert_almost_equal(v_rad2, v_rad2_mp, decimal=4)
        npt.assert_almost_equal(v_tang1, v_tang1_mp, decimal=4)
        npt.assert_almost_equal(v_tang2, v_tang2_mp, decimal=4)

    def test_radial_tangential_distortions_multi_plane(self):
        lens_model_list = ["SIS", "SHEAR"]
        lens_mp = LensModel(
            lens_model_list=lens_model_list,
            lens_redshift_list=[0.5, 0.4],
            multi_plane=True,
            z_source=2,
        )
        lens = LensModel(lens_model_list=lens_model_list)
        x0, y0 = 2.0, 1.0

        kwargs_lens = [
            {"theta_E": 1, "center_x": 0, "center_y": 0},
            {"gamma1": 0.0, "gamma2": 0.00001},
        ]

        extensions = LensModelExtensions(lensModel=lens)
        extensions_mp = LensModelExtensions(lensModel=lens_mp)

        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = extensions.radial_tangential_differentials(
            x=x0, y=y0, kwargs_lens=kwargs_lens, smoothing_3rd=0.0001
        )

        (
            lambda_rad_mp,
            lambda_tan_mp,
            orientation_angle_mp,
            dlambda_tan_dtan_mp,
            dlambda_tan_drad_mp,
            dlambda_rad_drad_mp,
            dlambda_rad_dtan_mp,
            dphi_tan_dtan_mp,
            dphi_tan_drad_mp,
            dphi_rad_drad_mp,
            dphi_rad_dtan_mp,
        ) = extensions_mp.radial_tangential_differentials(
            x=x0, y=y0, kwargs_lens=kwargs_lens, smoothing_3rd=0.0001
        )

        npt.assert_almost_equal(lambda_rad, lambda_rad_mp, decimal=3)
        npt.assert_almost_equal(lambda_tan, lambda_tan_mp, decimal=3)
        npt.assert_almost_equal(dphi_tan_dtan, dphi_rad_dtan_mp, decimal=3)

    def test_radial_tangential_differentials(self):
        from lenstronomy.Util import util

        x, y = util.make_grid(numPix=10, deltapix=1)

        lens_model_list = ["SIS"]
        center_x, center_y = 0, 0
        lens = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{"theta_E": 1, "center_x": center_x, "center_y": center_y}]

        extensions = LensModelExtensions(lensModel=lens)
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = extensions.radial_tangential_differentials(
            2, 2, kwargs_lens, smoothing_3rd=0.001
        )

        npt.assert_almost_equal(lambda_rad, 1, decimal=5)
        npt.assert_almost_equal(lambda_tan, 1.5469181606780271, decimal=5)

        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = extensions.radial_tangential_differentials(
            np.array([2]), np.array([2]), kwargs_lens, smoothing_3rd=0.001
        )

        npt.assert_almost_equal(lambda_rad, 1, decimal=5)
        npt.assert_almost_equal(lambda_tan, 1.5469181606780271, decimal=5)

        mag = lens.magnification(x, y, kwargs_lens)
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = extensions.radial_tangential_differentials(
            x, y, kwargs_lens, smoothing_3rd=0.001
        )
        mag_tang_rad = lambda_tan * lambda_rad
        npt.assert_almost_equal(mag_tang_rad, mag, decimal=5)

    def test_curved_arc_estimate(self):
        lens_model_list = ["SPP"]
        lens = LensModel(lens_model_list=lens_model_list)
        arc = LensModel(lens_model_list=["CURVED_ARC_SPP"])
        theta_E = 4
        gamma = 2.0
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        ext = LensModelExtensions(lensModel=lens)
        x_0, y_0 = 5, 0
        kwargs_arc = ext.curved_arc_estimate(x_0, y_0, kwargs_lens)
        (
            theta_E_arc,
            gamma_arc,
            center_x_spp_arc,
            center_y_spp_arc,
        ) = CurvedArcSPP.stretch2spp(**kwargs_arc)
        npt.assert_almost_equal(theta_E_arc, theta_E, decimal=4)
        npt.assert_almost_equal(gamma_arc, gamma, decimal=3)
        npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
        npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)
        x, y = util.make_grid(numPix=10, deltapix=1)
        alpha_x, alpha_y = lens.alpha(x, y, kwargs_lens)
        alpha0_x, alpha0_y = lens.alpha(x_0, y_0, kwargs_lens)
        alpha_x_arc, alpha_y_arc = arc.alpha(x, y, [kwargs_arc])
        npt.assert_almost_equal(alpha_x_arc, alpha_x - alpha0_x, decimal=3)
        npt.assert_almost_equal(alpha_y_arc, alpha_y - alpha0_y, decimal=3)

        x_0, y_0 = 0.0, 3
        kwargs_arc = ext.curved_arc_estimate(x_0, y_0, kwargs_lens)
        (
            theta_E_arc,
            gamma_arc,
            center_x_spp_arc,
            center_y_spp_arc,
        ) = CurvedArcSPP.stretch2spp(**kwargs_arc)
        print(kwargs_arc)
        print(theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc)
        npt.assert_almost_equal(theta_E_arc, theta_E, decimal=4)
        npt.assert_almost_equal(gamma_arc, gamma, decimal=3)
        npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
        npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)

        x_0, y_0 = -2, -3
        kwargs_arc = ext.curved_arc_estimate(x_0, y_0, kwargs_lens)
        (
            theta_E_arc,
            gamma_arc,
            center_x_spp_arc,
            center_y_spp_arc,
        ) = CurvedArcSPP.stretch2spp(**kwargs_arc)
        npt.assert_almost_equal(theta_E_arc, theta_E, decimal=4)
        npt.assert_almost_equal(gamma_arc, gamma, decimal=3)
        npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
        npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)

    def test_curved_arc_finite_area(self):
        lens_model_list = ["SPP"]
        lens = LensModel(lens_model_list=lens_model_list)
        arc = LensModel(lens_model_list=["CURVED_ARC_SPP"])
        theta_E = 4
        gamma = 2.0
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        ext = LensModelExtensions(lensModel=lens)
        x_0, y_0 = 5, 0
        kwargs_arc = ext.curved_arc_estimate(x_0, y_0, kwargs_lens)

        dr = 0.001
        kwargs_arc_finite = ext.curved_arc_finite_area(x_0, y_0, kwargs_lens, dr)
        npt.assert_almost_equal(
            kwargs_arc["direction"], kwargs_arc_finite["direction"], decimal=3
        )
        npt.assert_almost_equal(
            kwargs_arc["radial_stretch"], kwargs_arc_finite["radial_stretch"], decimal=3
        )
        npt.assert_almost_equal(
            kwargs_arc["tangential_stretch"],
            kwargs_arc_finite["tangential_stretch"],
            decimal=3,
        )
        npt.assert_almost_equal(
            kwargs_arc["curvature"], kwargs_arc_finite["curvature"], decimal=3
        )

    def test_curved_arc_estimate_tan_diff(self):
        arc_tan_diff = CurvedArcTanDiff()
        lens_model_list = ["SIE"]
        lens = LensModel(lens_model_list=lens_model_list)
        arc = LensModel(lens_model_list=["CURVED_ARC_TAN_DIFF"])
        theta_E = 4

        # here we model an off-axis ellisoid relative to the x-axis
        e1, e2 = 0.0, -0.1
        x_0, y_0 = 5, 0
        kwargs_lens = [
            {"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": 0, "center_y": 0}
        ]
        ext = LensModelExtensions(lensModel=lens)

        kwargs_arc = ext.curved_arc_estimate(
            x_0, y_0, kwargs_lens, tan_diff=True, smoothing_3rd=0.01
        )
        (
            theta_E_sie,
            e1_sie,
            e2_sie,
            kappa_ext,
            center_x_sis,
            center_y_sis,
        ) = arc_tan_diff.stretch2sie_mst(**kwargs_arc)
        print(theta_E_sie, e1_sie, e2_sie, center_x_sis, center_y_sis)
        npt.assert_almost_equal(e2_sie - e2, 0, decimal=1)
        npt.assert_almost_equal(e1_sie, e1, decimal=3)

        # here we model an off-axis ellisoid relative to the y-axis
        e1, e2 = 0.1, 0.0
        x_0, y_0 = 0, 5
        kwargs_lens = [
            {"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": 0, "center_y": 0}
        ]
        ext = LensModelExtensions(lensModel=lens)

        kwargs_arc = ext.curved_arc_estimate(
            x_0, y_0, kwargs_lens, tan_diff=True, smoothing_3rd=0.01
        )
        (
            theta_E_sie,
            e1_sie,
            e2_sie,
            kappa_ext,
            center_x_sis,
            center_y_sis,
        ) = arc_tan_diff.stretch2sie_mst(**kwargs_arc)
        print(theta_E_sie, e1_sie, e2_sie, center_x_sis, center_y_sis)
        npt.assert_almost_equal(e1_sie - e1, 0, decimal=1)
        npt.assert_almost_equal(e2_sie, e2, decimal=3)

        x, y = util.make_grid(numPix=100, deltapix=0.1)
        kappa = lens.kappa(x, y, kwargs_lens)
        kappa_arc = arc.kappa(x, y, [kwargs_arc])

    def test_arcs_at_image_position(self):
        # lensing quantities
        kwargs_spp = {
            "theta_E": 1.26,
            "gamma": 2.0,
            "e1": 0.1,
            "e2": -0.1,
            "center_x": 0.0,
            "center_y": 0.0,
        }  # parameters of the deflector lens model

        # the lens model is a supperposition of an elliptical lens model with external shear
        lens_model_list = ["SPEP"]  # , 'SHEAR']
        kwargs_lens = [kwargs_spp]  # , kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        lensEquationSolver = LensEquationSolver(lens_model_class)
        source_x = 0.0
        source_y = 0.05
        x_image, y_image = lensEquationSolver.findBrightImage(
            source_x,
            source_y,
            kwargs_lens,
            numImages=4,
            min_distance=0.05,
            search_window=5,
        )
        arc_model = LensModel(lens_model_list=["CURVED_ARC_SPP", "SHIFT"])
        for i in range(len(x_image)):
            x0, y0 = x_image[i], y_image[i]
            print(x0, y0, i)
            ext = LensModelExtensions(lensModel=lens_model_class)
            kwargs_arc_i = ext.curved_arc_estimate(x0, y0, kwargs_lens)
            alpha_x, alpha_y = lens_model_class.alpha(x0, y0, kwargs_lens)
            kwargs_arc = [kwargs_arc_i, {"alpha_x": alpha_x, "alpha_y": alpha_y}]
            print(kwargs_arc_i)
            direction = kwargs_arc_i["direction"]
            print(np.cos(direction), np.sin(direction))
            x, y = util.make_grid(numPix=5, deltapix=0.01)
            x = x0
            y = y0
            gamma1_arc, gamma2_arc = arc_model.gamma(x, y, kwargs_arc)
            gamma1, gamma2 = lens_model_class.gamma(x, y, kwargs_lens)
            print(gamma1, gamma2)
            npt.assert_almost_equal(gamma1_arc, gamma1, decimal=3)
            npt.assert_almost_equal(gamma2_arc, gamma2, decimal=3)
            (
                theta_E_arc,
                gamma_arc,
                center_x_spp_arc,
                center_y_spp_arc,
            ) = CurvedArcSPP.stretch2spp(**kwargs_arc_i)
            print(theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc)
            npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
            npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)

    def test_analytic_differentials_spp(self):
        """Test the analytical differentials of the spherical power-law mass profile.

        :return:
        """
        lens_model_class = LensModel(lens_model_list=["SPP"])
        theta_E = 2
        gamma = 2.2
        kwargs_lens = [
            {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
        ]
        ext = LensModelExtensions(lensModel=lens_model_class)
        # we leave out the very high magnification region where the numerical differentials might not be as accurate as required in the tests
        x = np.append(
            np.linspace(start=0.1, stop=theta_E - 0.1, num=9),
            np.linspace(start=theta_E + 0.1, stop=3 * theta_E, num=9),
        )
        y = np.zeros_like(x)
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(x, y, kwargs_lens)

        def _lambda_t_analytic(r, theta_E, gamma):
            """Analytic expression for lambda_tan.

            :param r: radius
            :return:
            """
            return (1 - (theta_E / r) ** (gamma - 1)) ** -1

        lambda_tan_analytic = _lambda_t_analytic(x, theta_E, gamma)
        npt.assert_almost_equal(lambda_tan_analytic, lambda_tan, decimal=5)

        def _lambda_r_analytic(r, theta_E, gamma):
            """Analytic expression for lambda_rad.

            :param r: radius
            :return:
            """
            return (1 + (gamma - 2) * (theta_E / r) ** (gamma - 1)) ** -1

        lambda_rad_analytic = _lambda_r_analytic(x, theta_E, gamma)
        npt.assert_almost_equal(lambda_rad_analytic, lambda_rad, decimal=5)

        def _lambda_t_dr_analytic(r, theta_E, gamma):
            """Analytic expression for  d(lambda_tan) / dr.

            :param r: radius
            :return:
            """
            return (
                (1 - gamma)
                * (theta_E / r) ** gamma
                / (theta_E * (1 - (theta_E / r) ** (gamma - 1)) ** 2)
            )

        dlambda_tan_drad_analytic = _lambda_t_dr_analytic(x, theta_E, gamma)
        npt.assert_almost_equal(
            dlambda_tan_drad_analytic / dlambda_tan_drad, 1, decimal=2
        )

        def _lambda_r_dr_analytic(r, theta_E, gamma):
            """Analytic expression for  d(lambda_tan) / dr.

            :param r: radius
            :return:
            """
            return (
                (1 - gamma)
                * (2 - gamma)
                * (theta_E / r) ** gamma
                / (theta_E * (1 + (theta_E / r) ** (gamma - 1) * (gamma - 2)) ** 2)
            )

        dlambda_r_drad_analytic = _lambda_r_dr_analytic(x, theta_E, gamma)
        npt.assert_almost_equal(
            dlambda_r_drad_analytic / dlambda_rad_drad, 1, decimal=2
        )

    def test_analytic_differential_pemd(self):
        lens_model_class = LensModel(lens_model_list=["EPL"])
        theta_E = 4
        gamma = 2.0

        ext = LensModelExtensions(lensModel=lens_model_class)
        # we leave out the very high magnification region where the numerical differentials might not be as accurate as required in the tests
        x = np.append(
            np.linspace(start=0.2 * theta_E, stop=theta_E * (1 - 0.2), num=9),
            np.linspace(start=theta_E * (1 + 0.5), stop=3 * theta_E, num=9),
        )
        y = np.zeros_like(x)

        def _dlambda_t_dr_analytic(r, theta_E, gamma):
            """Analytic expression for  d(lambda_tan) / dr.

            :param r: circularized radius
            :return:
            """
            return (
                (1 - gamma)
                * (theta_E / r) ** gamma
                / (theta_E * (1 - (theta_E / r) ** (gamma - 1)) ** 2)
            )

        def _dlambda_t_dt_analytic(x, y, theta_E, gamma, q, phi_G):
            # polar coordinates with respect to x-axis
            r, phi = param_util.cart2polar(x, y, center_x=0, center_y=0)
            epsilon = (1 - q**2) / (1 + q**2)

            # radial component in respect to rotation of deflector
            r_ = r * np.sqrt(1 - epsilon * np.cos(2 * (phi - phi_G)))
            # equivalent  Einstein radius for elliptical mass definition
            theta_E_prim = np.sqrt(2 * q / (1 + q**2)) * theta_E
            dlambda_t_dr = _dlambda_t_dr_analytic(r_, theta_E_prim, gamma)
            dr_de_t = (
                epsilon
                * np.sin(2 * (phi - phi_G))
                / np.sqrt(1 - epsilon * np.cos(2 * (phi - phi_G)))
            )
            return dlambda_t_dr * dr_de_t, dlambda_t_dr

        # define ellipticity with axis ratio and orientation

        # ellipticity aligned -> leading to zero differential
        phi_G, q = 0.0, 0.7
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(x, y, kwargs_lens)
        dlambda_tan_dtan_analytic, dlambda_t_dr_analtic = _dlambda_t_dt_analytic(
            x, y, theta_E, gamma, q, phi_G
        )
        npt.assert_almost_equal(dlambda_tan_dtan_analytic, 0, decimal=5)
        npt.assert_almost_equal(dlambda_tan_dtan, 0, decimal=1)

        # ellipticity aligned -> leading to zero differential
        phi_G, q = np.pi / 2, 0.7
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(x, y, kwargs_lens)
        dlambda_tan_dtan_analytic, dlambda_t_dr_analtic = _dlambda_t_dt_analytic(
            x, y, theta_E, gamma, q, phi_G
        )
        npt.assert_almost_equal(dlambda_tan_dtan_analytic, 0, decimal=5)
        npt.assert_almost_equal(dlambda_tan_dtan, 0, decimal=1)

        phi_G, q = np.pi / 4, 0.7
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(x, y, kwargs_lens)
        dlambda_tan_dtan_analytic, dlambda_t_dr_analtic = _dlambda_t_dt_analytic(
            x, y, theta_E, gamma, q, phi_G
        )
        print(dlambda_rad_drad, "dlambda_rad")
        npt.assert_almost_equal(dlambda_tan_drad / dlambda_t_dr_analtic, 1, decimal=2)
        print(dlambda_tan_dtan_analytic / dlambda_tan_dtan, "test")
        npt.assert_almost_equal(
            dlambda_tan_dtan_analytic / dlambda_tan_dtan, 1, decimal=2
        )

        phi_G, q = np.pi / 8, 0.7
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(x, y, kwargs_lens)
        dlambda_tan_dtan_analytic, dlambda_t_dr_analtic = _dlambda_t_dt_analytic(
            x, y, theta_E, gamma, q, phi_G
        )
        print(dlambda_tan_dtan_analytic / dlambda_tan_dtan, "test")
        npt.assert_almost_equal(
            dlambda_tan_dtan_analytic / dlambda_tan_dtan, 1, decimal=2
        )

        phi_G, q = 0.3, 0.7
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(x, y, kwargs_lens)
        dlambda_tan_dtan_analytic, dlambda_t_dr_analtic = _dlambda_t_dt_analytic(
            x, y, theta_E, gamma, q, phi_G
        )
        print(dlambda_tan_dtan_analytic / dlambda_tan_dtan, "test")
        npt.assert_almost_equal(
            dlambda_tan_dtan_analytic / dlambda_tan_dtan, 1, decimal=2
        )


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
