__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest
import unittest
from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane
from lenstronomy.LensModel.MultiPlane.multi_plane_base import MultiPlaneBase
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.MultiPlane.multi_plane import (
    LensedLocation,
    PhysicalLocation,
)
import lenstronomy.Util.constants as const


class TestMultiPlane(object):
    """Tests the source model routines."""

    def setup_method(self):
        pass

    def test_update_source_redshift(self):
        z_source = 1.5
        lens_model_list = ["SIS"]
        kwargs_lens = [{"theta_E": 1}]
        redshift_list = [0.5]
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            z_interp_stop=3,
            cosmo_interp=True,
        )
        alpha_x, alpha_y = lensModelMutli.alpha(1, 0, kwargs_lens=kwargs_lens)
        lensModelMutli.update_source_redshift(z_source=z_source)
        alpha_x_new, alpha_y_new = lensModelMutli.alpha(1, 0, kwargs_lens=kwargs_lens)
        npt.assert_almost_equal(alpha_x / alpha_x_new, 1.0, decimal=8)

        lensModelMutli.update_source_redshift(z_source=1.0)
        alpha_x_new, alpha_y_new = lensModelMutli.alpha(1, 0, kwargs_lens=kwargs_lens)
        assert alpha_x / alpha_x_new > 1

        lensModelMutli.update_source_redshift(z_source=2.0)
        alpha_x_new, alpha_y_new = lensModelMutli.alpha(1, 0, kwargs_lens=kwargs_lens)
        assert alpha_x / alpha_x_new < 1

    def test_sis_alpha(self):
        z_source = 1.5
        lens_model_list = ["SIS"]
        redshift_list = [0.5]
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        alpha_x_simple, alpha_y_simple = lensModel.alpha(1, 0, kwargs_lens)
        alpha_x_multi, alpha_y_multi = lensModelMutli.alpha(1, 0, kwargs_lens)
        npt.assert_almost_equal(alpha_x_simple, alpha_x_multi, decimal=8)
        npt.assert_almost_equal(alpha_y_simple, alpha_y_multi, decimal=8)
        sum_partial = (
            np.sum(lensModelMutli._multi_plane_base._T_ij_list)
            + lensModelMutli._T_ij_stop
        )
        T_z_true = lensModelMutli._T_z_source
        npt.assert_almost_equal(sum_partial, T_z_true, decimal=5)

    def test_sis_ray_tracing(self):
        z_source = 1.5
        lens_model_list = ["SIS"]
        redshift_list = [0.5]
        from astropy.cosmology import FlatLambdaCDM, LambdaCDM

        # test flat LCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            cosmo=cosmo,
        )
        lensModel = LensModel(lens_model_list=lens_model_list, cosmo=cosmo)
        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        beta_x_simple, beta_y_simple = lensModel.ray_shooting(1, 0, kwargs_lens)
        beta_x_multi, beta_y_multi = lensModelMutli.ray_shooting(1, 0, kwargs_lens)
        npt.assert_almost_equal(beta_x_simple, beta_x_multi, decimal=10)
        npt.assert_almost_equal(beta_y_simple, beta_y_multi, decimal=10)
        npt.assert_almost_equal(beta_x_simple, 0, decimal=10)
        npt.assert_almost_equal(beta_y_simple, 0, decimal=10)

    def test_sis_hessian(self):
        z_source = 1.5
        lens_model_list = ["SIS"]
        redshift_list = [0.5]
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        f_xx_simple, f_xy_simple, f_yx_simple, f_yy_simple = lensModel.hessian(
            1, 0, kwargs_lens
        )
        f_xx_multi, f_xy_multi, f_yx_multi, f_yy_multi = lensModelMutli.hessian(
            1, 0, kwargs_lens, diff=0.000001
        )
        npt.assert_almost_equal(f_xx_simple, f_xx_multi, decimal=5)
        npt.assert_almost_equal(f_xy_simple, f_xy_multi, decimal=5)
        npt.assert_almost_equal(f_yx_simple, f_yx_multi, decimal=5)
        npt.assert_almost_equal(f_yy_simple, f_yy_multi, decimal=5)

    def test_empty(self):
        z_source = 1.5
        lens_model_list = []
        redshift_list = []
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        kwargs_lens = []
        f_xx_multi, f_xy_multi, f_yx_multi, f_yy_multi = lensModelMutli.hessian(
            1, 0, kwargs_lens, diff=0.000001
        )
        npt.assert_almost_equal(0, f_xx_multi, decimal=5)
        npt.assert_almost_equal(0, f_xy_multi, decimal=5)
        npt.assert_almost_equal(0, f_yx_multi, decimal=5)
        npt.assert_almost_equal(0, f_yy_multi, decimal=5)

    def test_sis_kappa_gamma_mag(self):
        z_source = 1.5
        lens_model_list = ["SIS"]
        redshift_list = [0.5]
        lensModelMutli = LensModel(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            multi_plane=True,
        )
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        kappa_simple = lensModel.kappa(1, 0, kwargs_lens)
        kappa_multi = lensModelMutli.kappa(1, 0, kwargs_lens)
        npt.assert_almost_equal(kappa_simple, kappa_multi, decimal=5)

        gamma1_simple, gamma2_simple = lensModel.gamma(1, 0, kwargs_lens)
        gamma1_multi, gamma2_multi = lensModelMutli.gamma(1, 0, kwargs_lens)
        npt.assert_almost_equal(gamma1_simple, gamma1_multi, decimal=5)
        npt.assert_almost_equal(gamma2_simple, gamma2_multi, decimal=5)

        mag_simple = lensModel.magnification(0.99, 0, kwargs_lens)
        mag_multi = lensModelMutli.magnification(0.99, 0, kwargs_lens)
        npt.assert_almost_equal(mag_simple, mag_multi, decimal=5)

    def test_sis_travel_time(self):
        z_source = 1.5
        z_lens = 0.5
        lens_model_list = ["SIS"]
        redshift_list = [z_lens]
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{"theta_E": 1.0, "center_x": 0, "center_y": 0}]
        dt = lensModelMutli.arrival_time(1.0, 0.0, kwargs_lens)
        Dt = lensModelMutli._multi_plane_base._cosmo_bkg.ddt(
            z_lens=z_lens, z_source=z_source
        )
        fermat_pot = lensModel.fermat_potential(1, 0.0, kwargs_lens)
        dt_simple = const.delay_arcsec2days(fermat_pot, Dt)
        print(dt, dt_simple)
        npt.assert_almost_equal(dt, dt_simple, decimal=8)

    def test_sis_travel_time_new(self):
        z_source = 1.5
        z_lens = 0.5
        lens_model_list = ["SIS", "SIS"]
        redshift_list = [z_lens, 0.2]
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [
            {"theta_E": 1.0, "center_x": 0, "center_y": 0},
            {"theta_E": 0.0, "center_x": 0, "center_y": 0},
        ]
        dt = lensModelMutli.arrival_time(1.0, 0.0, kwargs_lens)
        Dt = lensModelMutli._multi_plane_base._cosmo_bkg.ddt(
            z_lens=z_lens, z_source=z_source
        )
        fermat_pot = lensModel.fermat_potential(1, 0.0, kwargs_lens)
        dt_simple = const.delay_arcsec2days(fermat_pot, Dt)
        print(dt, dt_simple)
        npt.assert_almost_equal(dt, dt_simple, decimal=8)

    def test_sis_ray_shooting(self):
        z_source = 1.5
        z_lens = 0.5
        lens_model_list = ["SIS"]
        redshift_list = [z_lens]
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{"theta_E": 1.0, "center_x": 0, "center_y": 0}]
        beta_x, beta_y = lensModelMutli.ray_shooting(1.0, 0.0, kwargs_lens)
        beta_x_single, beta_y_single = lensModel.ray_shooting(1, 0.0, kwargs_lens)
        npt.assert_almost_equal(beta_x, beta_x_single, decimal=8)
        npt.assert_almost_equal(beta_y, beta_y_single, decimal=8)
        x, y = np.array([1.0]), np.array([2.0])
        beta_x, beta_y = lensModelMutli.ray_shooting(x, y, kwargs_lens)
        beta_x_single, beta_y_single = lensModel.ray_shooting(x, y, kwargs_lens)
        npt.assert_almost_equal(beta_x, beta_x_single, decimal=8)
        npt.assert_almost_equal(beta_y, beta_y_single, decimal=8)

    def test_random_ordering(self):
        z_source = 1.5
        lens_model_list = ["SIS", "SIS", "SIS"]
        sis1 = {"theta_E": 1.0, "center_x": 0, "center_y": 0}
        sis2 = {"theta_E": 0.2, "center_x": 0.5, "center_y": 0}
        sis3 = {"theta_E": 0.1, "center_x": 0, "center_y": 0.5}
        z1 = 0.1
        z2 = 0.5
        z3 = 0.7
        redshift_list = [z1, z2, z3]
        kwargs_lens = [sis1, sis2, sis3]
        lensModel = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        beta_x_1, beta_y_1 = lensModel.ray_shooting(1.0, 0.0, kwargs_lens)

        redshift_list = [z3, z2, z1]
        kwargs_lens = [sis3, sis2, sis1]
        lensModel = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        beta_x_2, beta_y_2 = lensModel.ray_shooting(1.0, 0.0, kwargs_lens)
        npt.assert_almost_equal(beta_x_1, beta_x_2, decimal=8)
        npt.assert_almost_equal(beta_y_1, beta_y_2, decimal=8)

    def test_ray_shooting_partial_2(self):
        z_source = 1.5
        lens_model_list = ["SIS", "SIS", "SIS", "SIS"]
        sis1 = {"theta_E": 0.4, "center_x": 0, "center_y": 0}
        sis2 = {"theta_E": 0.2, "center_x": 0.5, "center_y": 0}
        sis3 = {"theta_E": 0.1, "center_x": 0, "center_y": 0.5}
        sis4 = {"theta_E": 0.5, "center_x": 0.1, "center_y": 0.3}

        lens_model_list_macro = ["SIS"]
        kwargs_macro = [{"theta_E": 1, "center_x": 0, "center_y": 0}]

        zmacro = 0.5

        z1 = 0.1
        z2 = 0.5
        z3 = 0.5
        z4 = 0.7
        redshift_list = [z1, z2, z3, z4]
        kwargs_lens = [sis1, sis2, sis3, sis4]
        kwargs_lens_full = kwargs_macro + kwargs_lens
        lensModel_full = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list_macro + lens_model_list,
            lens_redshift_list=[zmacro] + redshift_list,
        )
        lensModel_macro = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list_macro,
            lens_redshift_list=[zmacro],
        )
        lensModel = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )

        theta_x, theta_y = 1.0, 1.0

        (
            x_subs,
            y_subs,
            alpha_x_subs,
            alpha_y_subs,
        ) = lensModel.ray_shooting_partial_comoving(
            x=0,
            y=0,
            alpha_x=theta_x,
            alpha_y=theta_y,
            z_start=0,
            z_stop=zmacro,
            kwargs_lens=kwargs_lens,
        )

        (
            x_out,
            y_out,
            alpha_x_out,
            alpha_y_out,
        ) = lensModel_macro.ray_shooting_partial_comoving(
            x_subs,
            y_subs,
            alpha_x_subs,
            alpha_y_subs,
            zmacro,
            zmacro,
            kwargs_macro,
            include_z_start=True,
        )
        npt.assert_almost_equal(x_subs, x_out)
        npt.assert_almost_equal(y_subs, y_out)

        (
            x_full,
            y_full,
            alpha_x_full,
            alpha_y_full,
        ) = lensModel_full.ray_shooting_partial_comoving(
            0, 0, theta_x, theta_y, 0, zmacro, kwargs_lens_full
        )

        npt.assert_almost_equal(x_full, x_out)
        npt.assert_almost_equal(y_full, y_out)
        npt.assert_almost_equal(alpha_x_full, alpha_x_out)
        npt.assert_almost_equal(alpha_y_full, alpha_y_out)

        x_src, y_src, _, _ = lensModel_full.ray_shooting_partial_comoving(
            x=x_out,
            y=y_out,
            alpha_x=alpha_x_out,
            alpha_y=alpha_y_out,
            z_start=zmacro,
            z_stop=z_source,
            kwargs_lens=kwargs_lens_full,
        )

        beta_x, beta_y = lensModel.co_moving2angle_source(x_src, y_src)
        beta_x_true, beta_y_true = lensModel_full.ray_shooting(
            theta_x, theta_y, kwargs_lens_full
        )

        npt.assert_almost_equal(beta_x, beta_x_true, decimal=8)
        npt.assert_almost_equal(beta_y, beta_y_true, decimal=8)

    def test_ray_shooting_partial(self):
        z_source = 1.5
        lens_model_list = ["SIS", "SIS", "SIS"]
        sis1 = {"theta_E": 1.0, "center_x": 0, "center_y": 0}
        sis2 = {"theta_E": 0.2, "center_x": 0.5, "center_y": 0}
        sis3 = {"theta_E": 0.1, "center_x": 0, "center_y": 0.5}
        z1 = 0.1
        z2 = 0.5
        z3 = 0.7
        redshift_list = [z1, z2, z3]
        kwargs_lens = [sis1, sis2, sis3]
        lensModel = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        lensModel_2 = LensModel(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            multi_plane=True,
        )
        multiplane_2 = lensModel_2.lens_model
        intermediate_index = 1
        theta_x, theta_y = 1.0, 1.0

        Tzsrc = lensModel._multi_plane_base._cosmo_bkg.T_xy(0, z_source)

        z_intermediate = lensModel._multi_plane_base._lens_redshift_list[
            intermediate_index
        ]

        for lensmodel_class in [lensModel, multiplane_2]:
            (
                x_out,
                y_out,
                alpha_x_out,
                alpha_y_out,
            ) = lensmodel_class.ray_shooting_partial_comoving(
                x=0,
                y=0,
                alpha_x=theta_x,
                alpha_y=theta_y,
                z_start=0,
                z_stop=z_intermediate,
                kwargs_lens=kwargs_lens,
            )

            x_out_full_0 = x_out
            y_out_full_0 = y_out

            (
                x_out,
                y_out,
                alpha_x_out,
                alpha_y_out,
            ) = lensmodel_class.ray_shooting_partial_comoving(
                x=x_out,
                y=y_out,
                alpha_x=alpha_x_out,
                alpha_y=alpha_y_out,
                z_start=z_intermediate,
                z_stop=z_source,
                kwargs_lens=kwargs_lens,
            )

            beta_x, beta_y = lensModel.co_moving2angle_source(x_out, y_out)
            beta_x_true, beta_y_true = lensmodel_class.ray_shooting(
                theta_x, theta_y, kwargs_lens
            )
            npt.assert_almost_equal(beta_x, beta_x_true, decimal=8)
            npt.assert_almost_equal(beta_y, beta_y_true, decimal=8)

            T_ij_start = lensModel._multi_plane_base._cosmo_bkg.T_xy(
                z_observer=0, z_source=0.1
            )
            T_ij_end = lensModel._multi_plane_base._cosmo_bkg.T_xy(
                z_observer=0.7, z_source=1.5
            )
            (
                x_out,
                y_out,
                alpha_x_out,
                alpha_y_out,
            ) = lensmodel_class.ray_shooting_partial_comoving(
                x=0,
                y=0,
                alpha_x=theta_x,
                alpha_y=theta_y,
                z_start=0,
                z_stop=z_source,
                kwargs_lens=kwargs_lens,
                T_ij_start=T_ij_start,
                T_ij_end=T_ij_end,
            )

            beta_x, beta_y = x_out / Tzsrc, y_out / Tzsrc
            npt.assert_almost_equal(beta_x, beta_x_true, decimal=8)
            npt.assert_almost_equal(beta_y, beta_y_true, decimal=8)

    def test_pseudo_multiplane(self):
        z_source = 1.5
        lens_model_list = ["SIS", "SIS"]
        sis1 = {"theta_E": 1.0, "center_x": 0, "center_y": 0}
        sis2 = {"theta_E": 0.2, "center_x": 0.5, "center_y": 0}
        z1 = 0.5
        z2 = 0.5

        redshift_list = [z1, z2]
        kwargs_lens = [sis1, sis2]
        lensModelMulti = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
        )
        lensModelSingle = LensModel(lens_model_list=lens_model_list)

        beta_x, beta_y = lensModelMulti.ray_shooting(1, 1, kwargs_lens)
        beta_x_single, beta_y_single = lensModelSingle.ray_shooting(1, 1, kwargs_lens)
        npt.assert_almost_equal(beta_x, beta_x_single, decimal=10)
        npt.assert_almost_equal(beta_y, beta_y_single, decimal=10)

    def test_position_convention(self):
        lens_model_list = ["SIS", "SIS", "SIS", "SIS"]
        redshift_list = [0.5, 0.5, 0.9, 0.6]

        kwargs_lens = [
            {"theta_E": 1, "center_x": 0, "center_y": 0},
            {"theta_E": 0.4, "center_x": 0, "center_y": 0.2},
            {"theta_E": 1, "center_x": 1.8, "center_y": -0.4},
            {"theta_E": 0.41, "center_x": 1.0, "center_y": 0.7},
        ]

        index_list = [[2, 3], [3, 2]]

        # compute the physical position given lensed position, and check that lensing computations
        # using the two different conventions and sets of kwargs agree

        for index in index_list:
            lensModel_observed = LensModel(
                lens_model_list=lens_model_list,
                multi_plane=True,
                observed_convention_index=index,
                z_source=1.5,
                lens_redshift_list=redshift_list,
            )
            lensModel_physical = LensModel(
                lens_model_list=lens_model_list,
                multi_plane=True,
                z_source=1.5,
                lens_redshift_list=redshift_list,
            )

            multi = lensModel_observed.lens_model._multi_plane_base
            lensed, phys = LensedLocation(multi, index), PhysicalLocation()

            kwargs_lens_physical = lensModel_observed.lens_model._convention(
                kwargs_lens
            )

            kwargs_phys, kwargs_lensed = phys(kwargs_lens), lensed(kwargs_lens)

            for j, lensed_kwargs in enumerate(kwargs_lensed):
                for ki in lensed_kwargs.keys():
                    assert lensed_kwargs[ki] == kwargs_lens_physical[j][ki]
                    assert kwargs_phys[j][ki] == kwargs_lens[j][ki]

            fxx, fyy, fxy, fyx = lensModel_observed.hessian(0.5, 0.5, kwargs_lens)
            fxx2, fyy2, fxy2, fyx2 = lensModel_physical.hessian(
                0.5, 0.5, kwargs_lens_physical
            )
            npt.assert_almost_equal(fxx, fxx2)
            npt.assert_almost_equal(fxy, fxy2)

            betax1, betay1 = lensModel_observed.ray_shooting(0.5, 0.5, kwargs_lens)
            betax2, betay2 = lensModel_physical.ray_shooting(
                0.5, 0.5, kwargs_lens_physical
            )
            npt.assert_almost_equal(betax1, betax2)
            npt.assert_almost_equal(betay1, betay2)

    def test_co_moving2angle_z1_z2(self):
        z_source = 1.5
        lens_model_list = ["SIS"]
        kwargs_lens = [{"theta_E": 1}]
        redshift_list = [0.5]
        lensModelMutli = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            z_interp_stop=3,
            cosmo_interp=False,
        )
        z1 = 0.5
        z2 = 1.5
        x, y = 1, 0
        beta_x, beta_y = lensModelMutli.co_moving2angle_z1_z2(x=x, y=y, z1=z1, z2=z2)
        T_ij_end = lensModelMutli._multi_plane_base._cosmo_bkg.T_xy(
            z_observer=z1, z_source=z2
        )
        npt.assert_almost_equal(beta_x, x / T_ij_end, decimal=7)
        npt.assert_almost_equal(beta_y, y / T_ij_end, decimal=7)

    def test_hessian_z1z2(self):
        z_source = 1.5
        lens_model_list = ["SIS"]
        kwargs_lens = [{"theta_E": 1}]
        redshift_list = [0.5]
        multi_plane = MultiPlane(
            z_source=z_source,
            lens_model_list=lens_model_list,
            lens_redshift_list=redshift_list,
            z_interp_stop=3,
            cosmo_interp=False,
        )
        z1, z2 = 0.000001, z_source
        theta_x, theta_y = np.linspace(start=-1, stop=1, num=10), np.linspace(
            start=-1, stop=1, num=10
        )
        f_xx_z12, f_xy_z12, f_yx_z12, f_yy_z12 = multi_plane.hessian_z1z2(
            z1, z2, theta_x, theta_y, kwargs_lens, diff=0.00000001
        )
        f_xx, f_xy, f_yx, f_yy = multi_plane.hessian(
            theta_x, theta_y, kwargs_lens, diff=0.00000001
        )
        npt.assert_almost_equal(f_xx_z12, f_xx, decimal=5)
        npt.assert_almost_equal(f_xy_z12, f_xy, decimal=5)
        npt.assert_almost_equal(f_yx_z12, f_yx, decimal=5)
        npt.assert_almost_equal(f_yy_z12, f_yy, decimal=5)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            MultiPlaneBase(
                z_source_convention=1, lens_model_list=["SIS"], lens_redshift_list=[2]
            )

        with self.assertRaises(ValueError):
            MultiPlaneBase(
                z_source_convention=1,
                lens_model_list=["SIS", "SIS"],
                lens_redshift_list=[0.5],
            )

        with self.assertRaises(ValueError):
            lens = MultiPlane(
                z_source_convention=1,
                z_source=1,
                lens_model_list=["SIS", "SIS"],
                lens_redshift_list=[0.5, 0.8],
            )
            lens._check_raise(k=[1])

        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["SIS"],
                multi_plane=True,
                z_source=1,
                z_source_convention=1,
                cosmo_interp=True,
                z_interp_stop=0.5,
            )


if __name__ == "__main__":
    pytest.main("-k TestLensModel")
