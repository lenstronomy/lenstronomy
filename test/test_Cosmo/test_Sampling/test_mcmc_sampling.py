import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Cosmo.Sampling.mcmc_sampling import MCMCSampler
from astropy.cosmology import FlatLambdaCDM


class TestMCMCSampling(object):

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

    def test_mcmc_emcee(self):
        n_walkers = 6
        n_run = 2
        n_burn = 2
        kwargs_mean_start = {'h0': self.H0_true}
        kwargs_fixed = {'om': self.omega_m_true}
        kwargs_sigma_start = {'h0': 5}
        kwargs_lower = {'h0': 10}
        kwargs_upper = {'h0': 200}
        kwargs_lens_list = [{'z_lens': self.z_L, 'z_source': self.z_S, 'likelihood_type': 'TDKin',
                             'D_d_sample': self.D_d_samples, 'D_delta_t_sample': self.D_dt_samples,
                             'kde_type': 'scipy_gaussian', 'bandwidth': 1}]
        cosmology = 'FLCDM'
        mcmc_sampler = MCMCSampler(kwargs_lens_list, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=False)
        samples = mcmc_sampler.mcmc_emcee(n_walkers, n_burn, n_run, kwargs_mean_start, kwargs_sigma_start)
        assert len(samples) == n_walkers*n_run

        name_list = mcmc_sampler.param_names(latex_style=False)
        assert len(name_list) == 1


if __name__ == '__main__':
    pytest.main()
