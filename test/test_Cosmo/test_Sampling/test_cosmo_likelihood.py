import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Cosmo.Sampling.cosmo_likelihood import CosmoLikelihood
from astropy.cosmology import FlatLambdaCDM


class TestCosmoLikelihood(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0

        self.H0_true = 70
        self.omega_m_true = 0.3
        self.cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.05)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=self.cosmo)
        self.Dd_true = lensCosmo.D_d
        self.D_dt_true = lensCosmo.D_dt

        self.sigma_Dd = 100
        self.sigma_Ddt = 100
        num_samples = 10000
        self.D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        self.D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)

        self.kwargs_lens_list = [{'z_lens': self.z_L, 'z_source': self.z_S, 'D_d_sample': self.D_d_samples,
                         'D_delta_t_sample': self.D_dt_samples, 'kde_type': 'scipy_gaussian', 'bandwidth': 1}]

    def test_log_likelihood(self):
        kwargs_lower = {'h0': 10, 'om': 0., 'ok': -0.8, 'w': -2, 'wa': -1, 'w0': -2, 'gamma_ppn': 0}
        kwargs_upper = {'h0': 200, 'om': 1, 'ok': 0.8, 'w': 0, 'wa': 1, 'w0': 1, 'gamma_ppn': 5}
        cosmology = 'oLCDM'
        cosmoL = CosmoLikelihood(self.kwargs_lens_list, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed={}, ppn_sampling=False)
        kwargs = {'h0': self.H0_true, 'om': self.omega_m_true, 'ok': 0}
        args = cosmoL._param.kwargs2args(kwargs)
        logl = cosmoL.likelihood(args=args)

        kwargs = {'h0': self.H0_true*0.99, 'om': self.omega_m_true, 'ok': 0}
        args = cosmoL._param.kwargs2args(kwargs)
        logl_sigma = cosmoL.likelihood(args=args)
        print(logl)
        npt.assert_almost_equal(logl - logl_sigma, 0.12, decimal=2)

        kwargs = {'h0': 100, 'om': 1., 'ok': 0.1}
        args = cosmoL._param.kwargs2args(kwargs)
        logl = cosmoL.likelihood(args=args)
        assert logl == -np.inf

        kwargs = {'h0': 100, 'om': .3, 'ok': -0.5}
        args = cosmoL._param.kwargs2args(kwargs)
        logl = cosmoL.likelihood(args=args)
        assert logl == -np.inf


if __name__ == '__main__':
    pytest.main()
