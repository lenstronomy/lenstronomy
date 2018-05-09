import numpy as np
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Cosmo.cosmography import KDELikelihood, MCMC_sampler


class TestCosmography(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0
        from astropy.cosmology import FlatLambdaCDM
        self.H0_true = 70
        self.omega_m_true = 0.3
        cosmo = FlatLambdaCDM(H0=self.H0_true, Om0=self.omega_m_true, Ob0=0.05)
        lensCosmo = LensCosmo(self.z_L, self.z_S, cosmo=cosmo)
        self.Dd_true = lensCosmo.D_d
        self.D_dt_true = lensCosmo.D_dt

        self.sigma_Dd = 100
        self.sigma_Ddt = 100
        num_samples = 10000
        self.D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        self.D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)

    def test_kde_likelihood(self):
        kdeLikelihood = KDELikelihood(self.D_d_samples, self.D_dt_samples)
        logL_max = kdeLikelihood.logLikelihood(self.Dd_true, self.D_dt_true)
        logL_sigma = kdeLikelihood.logLikelihood(self.Dd_true+self.sigma_Dd, self.D_dt_true+self.sigma_Ddt)
        delta_log = logL_max - logL_sigma
        npt.assert_almost_equal(delta_log, 1, decimal=1)

    def test_sampling_H0_only(self):
        mcmc_sampler = MCMC_sampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_only",
                                    omega_m_fixed=self.omega_m_true, omega_mh2_fixed=self.omega_m_true*(self.H0_true/100)**2)
        walkerRatio = 10
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true]
        sigma_start = [5]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples)
        npt.assert_almost_equal(H0_mean/self.H0_true, 1, decimal=1)
        sigma = np.sqrt(np.var(mcmc_samples))
        npt.assert_almost_equal(sigma, 1.5, decimal=0)

    def test_sampling_H0_omega_m(self):
        mcmc_sampler = MCMC_sampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_omega_m",
                                    omega_m_fixed=self.omega_m_true, omega_mh2_fixed=self.omega_m_true*(self.H0_true/100)**2)
        walkerRatio = 10
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true, self.omega_m_true]
        sigma_start = [5, 0.1]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples[:, 0])
        npt.assert_almost_equal(H0_mean/self.H0_true, 1, decimal=1)
        sigma = np.sqrt(np.var(mcmc_samples[:, 0]))
        npt.assert_almost_equal(sigma, 2, decimal=0)

    def test_sampling_fix_omega_mh2(self):
        mcmc_sampler = MCMC_sampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="fix_omega_mh2",
                                    omega_m_fixed=self.omega_m_true,
                                    omega_mh2_fixed=self.omega_m_true * (self.H0_true / 100) ** 2)
        walkerRatio = 10
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true]
        sigma_start = [5]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1,
                                            init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples)
        npt.assert_almost_equal(H0_mean / self.H0_true, 1, decimal=1)
        sigma = np.sqrt(np.var(mcmc_samples))
        npt.assert_almost_equal(sigma, 1.5, decimal=0)

    def test_sampling_H0_omega_m_sklearn(self):
        mcmc_sampler = MCMC_sampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_omega_m",
                                    omega_m_fixed=self.omega_m_true, omega_mh2_fixed=self.omega_m_true*(self.H0_true/100)**2,
                                    kde_type='gaussian', bandwidth=10)
        walkerRatio = 10
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true, self.omega_m_true]
        sigma_start = [5, 0.1]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples[:, 0])
        npt.assert_almost_equal(H0_mean/self.H0_true, 1, decimal=1)
        sigma = np.sqrt(np.var(mcmc_samples[:, 0]))
        print(sigma)
        npt.assert_almost_equal(sigma, 2.6, decimal=0)