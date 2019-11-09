__author__ = 'sibirrer'

import numpy.testing as npt
import pytest

from lenstronomy.Analysis.td_cosmography import TDCosmography
import lenstronomy.Util.param_util as param_util


class TestTDCosmography(object):

    def setup(self):
        pass
"""

    def test_time_delays(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SPEP'], 'point_source_model_list': ['LENSED_POSITION']}
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.7)
        kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'e1': e1, 'e2': e2}]
        kwargs_else = [{'ra_image': [-1, 0, 1], 'dec_image': [0, 0, 0]}]
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        lensProp = KinematicAPI(z_lens, z_source, kwargs_options, cosmo=cosmo)
        delays = lensProp.time_delays(kwargs_lens, kwargs_ps=kwargs_else, kappa_ext=0)
        npt.assert_almost_equal(delays[0], -31.387590264501007, decimal=8)
        npt.assert_almost_equal(delays[1], 0, decimal=8)
        npt.assert_almost_equal(delays[2], -31.387590264501007, decimal=8)
        kappa_ext = 0.1
        delays_kappa = lensProp.time_delays(kwargs_lens, kwargs_ps=kwargs_else, kappa_ext=kappa_ext)
        npt.assert_almost_equal(delays_kappa/(1.-kappa_ext), delays, decimal=8)

        kappa_ext = 0.1
        delays_kappa = lensProp.time_delays(kwargs_lens, kwargs_ps=kwargs_else, kappa_ext=kappa_ext)
        npt.assert_almost_equal(delays_kappa / (1. - kappa_ext), delays, decimal=8)

    def test_angular_diameter_relations(self):
        z_lens = 0.5
        z_source = 1.5
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        lensProp = KinematicAPI(z_lens, z_source, kwargs_model={}, cosmo=cosmo)
        sigma_v_model = 290
        sigma_v = 310
        kappa_ext = 0
        D_dt_model = 3000

        D_d, Ds_Dds = lensProp.angular_diameter_relations(sigma_v_model, sigma_v, kappa_ext, D_dt_model)
        npt.assert_almost_equal(D_d, 992.768, decimal=1)
        npt.assert_almost_equal(Ds_Dds, 2.01, decimal=2)

    def test_angular_distances(self):
        z_lens = 0.5
        z_source = 1.5
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        lensProp = KinematicAPI(z_lens, z_source, kwargs_model={}, cosmo=cosmo)
        sigma_v_measured = 290
        sigma_v_modeled = 310
        kappa_ext = 0
        time_delay_measured = 111
        fermat_pot = 0.7
        Ds_Dds, DdDs_Dds = lensProp.angular_distances(sigma_v_measured, time_delay_measured, kappa_ext, sigma_v_modeled,
                                                      fermat_pot)
        print(Ds_Dds, DdDs_Dds)
        npt.assert_almost_equal(Ds_Dds, 1.5428, decimal=2)
        npt.assert_almost_equal(DdDs_Dds, 3775.442, decimal=1)
"""

if __name__ == '__main__':
    pytest.main()
