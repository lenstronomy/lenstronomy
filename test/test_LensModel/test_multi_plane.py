__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.multi_plane import MultiPlane
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.constants as const


class TestMultiPlane(object):
    """
    tests the source model routines
    """
    def setup(self):
        pass

    def test_sis_alpha(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        alpha_x_simple, alpha_y_simple = lensModel.alpha(1, 0, kwargs_lens)
        alpha_x_multi, alpha_y_multi = lensModelMutli.alpha(1, 0, kwargs_lens)
        assert alpha_x_simple == alpha_x_multi
        assert alpha_y_simple == alpha_y_multi
        sum_partial = np.sum(lensModelMutli._T_ij_list)
        T_z_true = lensModelMutli._T_z_source
        npt.assert_almost_equal(sum_partial, T_z_true, decimal=5)

    def test_sis_ray_tracing(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        beta_x_simple, beta_y_simple = lensModel.ray_shooting(1, 0, kwargs_lens)
        beta_x_multi, beta_y_multi = lensModelMutli.ray_shooting(1, 0, kwargs_lens)
        npt.assert_almost_equal(beta_x_simple, beta_x_multi, decimal=10)
        npt.assert_almost_equal(beta_y_simple, beta_y_multi, decimal=10)
        npt.assert_almost_equal(beta_x_simple, 0, decimal=10)
        npt.assert_almost_equal(beta_y_simple, 0, decimal=10)

    def test_sis_hessian(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        f_xx_simple, f_xy_simple, f_yx_simple, f_yy_simple = lensModel.hessian(1, 0, kwargs_lens)
        f_xx_multi, f_xy_multi, f_yx_multi, f_yy_multi = lensModelMutli.hessian(1, 0, kwargs_lens, diff=0.000001)
        npt.assert_almost_equal(f_xx_simple, f_xx_multi, decimal=5)
        npt.assert_almost_equal(f_xy_simple, f_xy_multi, decimal=5)
        npt.assert_almost_equal(f_yx_simple, f_yx_multi, decimal=5)
        npt.assert_almost_equal(f_yy_simple, f_yy_multi, decimal=5)

    def test_empty(self):
        z_source = 1.5
        lens_model_list = []
        redshift_list = []
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        kwargs_lens = []
        f_xx_multi, f_xy_multi, f_yx_multi, f_yy_multi = lensModelMutli.hessian(1, 0, kwargs_lens, diff=0.000001)
        npt.assert_almost_equal(0, f_xx_multi, decimal=5)
        npt.assert_almost_equal(0, f_xy_multi, decimal=5)
        npt.assert_almost_equal(0, f_yx_multi, decimal=5)
        npt.assert_almost_equal(0, f_yy_multi, decimal=5)

    def test_sis_kappa_gamma_mag(self):
        z_source = 1.5
        lens_model_list = ['SIS']
        redshift_list = [0.5]
        lensModelMutli = LensModel(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list, multi_plane=True)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
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
        lens_model_list = ['SIS']
        redshift_list = [z_lens]
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0}]
        dt = lensModelMutli.arrival_time(1., 0., kwargs_lens)
        Dt = lensModelMutli._cosmo_bkg.D_dt(z_lens=z_lens, z_source=z_source)
        fermat_pot = lensModel.fermat_potential(1, 0., 0., 0., kwargs_lens)
        dt_simple = const.delay_arcsec2days(fermat_pot, Dt)
        print(dt, dt_simple)
        npt.assert_almost_equal(dt, dt_simple, decimal=8)

    def test_sis_travel_time_new(self):
        z_source = 1.5
        z_lens = 0.5
        lens_model_list = ['SIS', 'SIS']
        redshift_list = [z_lens, 0.2]
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0}, {'theta_E': 0., 'center_x': 0, 'center_y': 0}]
        dt = lensModelMutli.arrival_time(1., 0., kwargs_lens)
        Dt = lensModelMutli._cosmo_bkg.D_dt(z_lens=z_lens, z_source=z_source)
        fermat_pot = lensModel.fermat_potential(1, 0., 0., 0., kwargs_lens)
        dt_simple = const.delay_arcsec2days(fermat_pot, Dt)
        print(dt, dt_simple)
        npt.assert_almost_equal(dt, dt_simple, decimal=8)

    def test_sis_ray_shooting(self):
        z_source = 1.5
        z_lens = 0.5
        lens_model_list = ['SIS']
        redshift_list = [z_lens]
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        lensModel = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0}]
        beta_x, beta_y = lensModelMutli.ray_shooting(1., 0., kwargs_lens)
        beta_x_single, beta_y_single = lensModel.ray_shooting(1, 0., kwargs_lens)
        npt.assert_almost_equal(beta_x, beta_x_single, decimal=8)
        npt.assert_almost_equal(beta_y, beta_y_single, decimal=8)
        x, y = np.array([1.]), np.array([2.])
        beta_x, beta_y = lensModelMutli.ray_shooting(x, y, kwargs_lens)
        beta_x_single, beta_y_single = lensModel.ray_shooting(x, y, kwargs_lens)
        npt.assert_almost_equal(beta_x, beta_x_single, decimal=8)
        npt.assert_almost_equal(beta_y, beta_y_single, decimal=8)

    def test_random_ordering(self):
        z_source = 1.5
        lens_model_list = ['SIS', 'SIS', 'SIS']
        sis1 = {'theta_E': 1., 'center_x': 0, 'center_y': 0}
        sis2 = {'theta_E': .2, 'center_x': 0.5, 'center_y': 0}
        sis3 = {'theta_E': .1, 'center_x': 0, 'center_y': 0.5}
        z1 = 0.1
        z2 = 0.5
        z3 = 0.7
        redshift_list = [z1, z2, z3]
        kwargs_lens = [sis1, sis2, sis3]
        lensModel = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        beta_x_1, beta_y_1 = lensModel.ray_shooting(1., 0., kwargs_lens)

        redshift_list = [z3, z2, z1]
        kwargs_lens = [sis3, sis2, sis1]
        lensModel = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        beta_x_2, beta_y_2 = lensModel.ray_shooting(1., 0., kwargs_lens)
        npt.assert_almost_equal(beta_x_1, beta_x_2, decimal=8)
        npt.assert_almost_equal(beta_y_1, beta_y_2, decimal=8)

    def test_ray_shooting_partial(self):
        z_source = 1.5
        lens_model_list = ['SIS', 'SIS', 'SIS']
        sis1 = {'theta_E': 1., 'center_x': 0, 'center_y': 0}
        sis2 = {'theta_E': .2, 'center_x': 0.5, 'center_y': 0}
        sis3 = {'theta_E': .1, 'center_x': 0, 'center_y': 0.5}
        z1 = 0.1
        z2 = 0.5
        z3 = 0.7
        redshift_list = [z1, z2, z3]
        kwargs_lens = [sis1, sis2, sis3]
        lensModel = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        z_intermediate = .5
        theta_x, theta_y = 1., 1.
        x_out, y_out, alpha_x_out, alpha_y_out = lensModel.ray_shooting_partial(x=0, y=0, alpha_x=theta_x,
                                            alpha_y=theta_y, z_start=0, z_stop=z_intermediate, kwargs_lens=kwargs_lens)
        x_out, y_out, alpha_x_out, alpha_y_out = lensModel.ray_shooting_partial(x=x_out, y=y_out, alpha_x=alpha_x_out,
                                                                            alpha_y=alpha_y_out, z_start=z_intermediate,
                                                                                z_stop=z_source,
                                                                                kwargs_lens=kwargs_lens)
        beta_x, beta_y = lensModel._co_moving2angle_source(x_out, y_out)
        beta_x_true, beta_y_true = lensModel.ray_shooting(theta_x, theta_y, kwargs_lens)
        npt.assert_almost_equal(beta_x, beta_x_true, decimal=8)
        npt.assert_almost_equal(beta_y, beta_y_true, decimal=8)
        x_out, y_out, alpha_x_out, alpha_y_out = lensModel.ray_shooting_partial(x=0, y=0, alpha_x=theta_x,
                                            alpha_y=theta_y, z_start=0, z_stop=z_source, kwargs_lens=kwargs_lens,
                                                                                keep_range=True)
        beta_x, beta_y = lensModel._co_moving2angle_source(x_out, y_out)
        npt.assert_almost_equal(beta_x, beta_x_true, decimal=8)
        npt.assert_almost_equal(beta_y, beta_y_true, decimal=8)


class TestForegroundShear(object):

    def setup(self):
        pass

    def test_foreground_shear(self):
        """
        scenario: a shear field in the foreground of the main deflector is placed
        we compute the expected shear on the lens plain and effectively model the same system in a single plane
        configuration
        We check for consistency of the two approaches and whether the specific redshift of the foreground shear field has
        an impact on the arrival time surface
        :return:
        """
        z_source = 1.5
        z_lens = 0.5
        z_shear = 0.2
        x, y = np.array([1., 0.]), np.array([0., 2.])
        from astropy.cosmology import default_cosmology
        from lenstronomy.Cosmo.background import Background

        cosmo = default_cosmology.get()
        cosmo_bkg = Background(cosmo)
        e1, e2 = 0.01, 0.01 # shear terms caused by z_shear on z_source
        lens_model_list = ['SIS', 'SHEAR']
        redshift_list = [z_lens, z_shear]
        lensModelMutli = MultiPlane(z_source=z_source, lens_model_list=lens_model_list, redshift_list=redshift_list)
        kwargs_lens_multi = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}, {'e1': e1, 'e2': e2}]
        alpha_x_multi, alpha_y_multi = lensModelMutli.alpha(x, y, kwargs_lens_multi)
        t_multi = lensModelMutli.arrival_time(x, y, kwargs_lens_multi)
        dt_multi = t_multi[0] - t_multi[1]
        physical_shear = cosmo_bkg.D_xy(0, z_source) / cosmo_bkg.D_xy(z_shear, z_source)
        foreground_factor = cosmo_bkg.D_xy(z_shear, z_lens) / cosmo_bkg.D_xy(0, z_lens) * physical_shear
        print(foreground_factor)
        lens_model_simple_list = ['SIS', 'FOREGROUND_SHEAR', 'SHEAR']
        kwargs_lens_single = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}, {'e1': e1*foreground_factor, 'e2': e2*foreground_factor}, {'e1': e1, 'e2': e2}]
        lensModel = LensModel(lens_model_list=lens_model_simple_list)
        alpha_x_simple, alpha_y_simple = lensModel.alpha(x, y, kwargs_lens_single)
        npt.assert_almost_equal(alpha_x_simple, alpha_x_multi, decimal=8)
        npt.assert_almost_equal(alpha_y_simple, alpha_y_multi, decimal=8)

        ra_source, dec_source = lensModel.ray_shooting(x, y, kwargs_lens_single)
        ra_source_multi, dec_source_multi = lensModelMutli.ray_shooting(x, y, kwargs_lens_multi)
        npt.assert_almost_equal(ra_source, ra_source_multi, decimal=8)
        npt.assert_almost_equal(dec_source, dec_source_multi, decimal=8)

        fermat_pot = lensModel.fermat_potential(x, y, ra_source, dec_source, kwargs_lens_single)
        from lenstronomy.Cosmo.lens_cosmo import LensCosmo
        lensCosmo = LensCosmo(z_lens, z_source, cosmo=cosmo)
        Dt = lensCosmo.D_dt
        print(lensCosmo.D_dt)
        #t_simple = const.delay_arcsec2days(fermat_pot, Dt)
        t_simple = lensCosmo.time_delay_units(fermat_pot)
        dt_simple = t_simple[0] - t_simple[1]
        print(t_simple, t_multi)
        npt.assert_almost_equal(dt_simple / dt_multi, 1, decimal=2)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
