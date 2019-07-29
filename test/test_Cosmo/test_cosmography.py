import numpy as np
import pytest
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Cosmo.cosmography import KDELikelihood, MCMCSampler, CosmoLikelihood, CosmoParam
from astropy.cosmology import FlatLambdaCDM
import unittest


class TestCosmography(object):

    def setup(self):
        np.random.seed(seed=41)
        self.z_L = 0.8
        self.z_S = 3.0

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

    def test_mcmc_emcee(self):
        n_walkers = 6
        n_run = 2
        n_burn = 2
        mean_start = [self.H0_true]
        sigma_start = [5]
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_only",
                                   omega_m_fixed=self.omega_m_true,
                                   omega_mh2_fixed=self.omega_m_true * (self.H0_true / 100) ** 2)
        samples = mcmc_sampler.mcmc_emcee(n_walkers, n_run, n_burn, mean_start, sigma_start)
        assert len(samples) == n_walkers*n_run

        mean_start = [self.H0_true, self.omega_m_true]
        sigma_start = [5, 0.1]
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_omega_m",
                                   omega_m_fixed=self.omega_m_true,
                                   omega_mh2_fixed=self.omega_m_true * (self.H0_true / 100) ** 2)
        samples = mcmc_sampler.mcmc_emcee(n_walkers, n_run, n_burn, mean_start, sigma_start)
        assert len(samples) == n_walkers * n_run

        mean_start = [self.H0_true]
        sigma_start = [5]
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="fix_omega_mh2",
                                   omega_m_fixed=self.omega_m_true,
                                   omega_mh2_fixed=self.omega_m_true * (self.H0_true / 100) ** 2)
        samples = mcmc_sampler.mcmc_emcee(n_walkers, n_run, n_burn, mean_start, sigma_start)
        assert len(samples) == n_walkers * n_run

        mean_start = [self.H0_true, self.omega_m_true, 1 - self.omega_m_true]
        sigma_start = [5, 0.1, 0.1]
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples,
                                   sampling_option="H0_omega_m_omega_de",
                                   omega_m_fixed=self.omega_m_true,
                                   omega_mh2_fixed=self.omega_m_true * (self.H0_true / 100) ** 2)
        samples = mcmc_sampler.mcmc_emcee(n_walkers, n_run, n_burn, mean_start, sigma_start)
        assert len(samples) == n_walkers * n_run

    def test_sampling_H0_only(self):
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_only",
                                   omega_m_fixed=self.omega_m_true, omega_mh2_fixed=self.omega_m_true*(self.H0_true/100)**2)
        walkerRatio = 10
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true]
        sigma_start = [2]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples)
        npt.assert_almost_equal(H0_mean/self.H0_true, 1, decimal=1)
        sigma = np.sqrt(np.var(mcmc_samples))
        npt.assert_almost_equal(sigma, 1.5, decimal=0)

    def test_sampling_H0_omega_m(self):
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_omega_m",
                                   omega_m_fixed=self.omega_m_true, omega_mh2_fixed=self.omega_m_true*(self.H0_true/100)**2)
        walkerRatio = 10
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true, self.omega_m_true]
        sigma_start = [5, 0.1]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples[:, 0])
        npt.assert_almost_equal(H0_mean/self.H0_true, 1, decimal=1)

    def test_sampling_fix_omega_mh2(self):
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="fix_omega_mh2",
                                   omega_m_fixed=self.omega_m_true,
                                   omega_mh2_fixed=self.omega_m_true * (self.H0_true / 100) ** 2)
        walkerRatio = 10
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true]
        sigma_start = [2]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1,
                                            init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples)
        npt.assert_almost_equal(H0_mean / self.H0_true, 1, decimal=1)
        sigma = np.sqrt(np.var(mcmc_samples))
        npt.assert_almost_equal(sigma, 1.5, decimal=0)

    def test_sampling_curvature(self):
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_omega_m_omega_de",
                                   omega_m_fixed=self.omega_m_true,
                                   omega_mh2_fixed=self.omega_m_true * (self.H0_true / 100) ** 2)
        walkerRatio = 4
        n_run = 10
        n_burn = 10
        mean_start = [self.H0_true, self.omega_m_true, 1 - self.omega_m_true]
        sigma_start = [5, 0.1, 0.1]
        mcmc_samples = mcmc_sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1,
                                            init_pos=None, mpi_monch=False)
        H0_mean = np.mean(mcmc_samples[:, 0])
        npt.assert_almost_equal(H0_mean / self.H0_true, 1, decimal=1)
        Om0_mean = np.mean(mcmc_samples[:, 1])
        npt.assert_almost_equal(Om0_mean / self.omega_m_true, 1, decimal=0)

    def test_sampling_H0_omega_m_sklearn(self):
        mcmc_sampler = MCMCSampler(self.z_L, self.z_S, self.D_d_samples, self.D_dt_samples, sampling_option="H0_omega_m",
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


class TestCosmoLikelihood(object):

    def setup(self):
        self.sigma_Dd = 100
        self.sigma_Ddt = 100
        num_samples = 100
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        z_L = 0.3
        z_S = 2
        lensCosmo = LensCosmo(z_L, z_S, cosmo=cosmo)
        self.Dd_true = lensCosmo.D_d
        self.D_dt_true = lensCosmo.D_dt
        D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)
        self.cosmoL = CosmoLikelihood(z_L, z_S, D_d_samples, D_dt_samples, sampling_option="H0_only", omega_m_fixed=0.3,
                                 omega_lambda_fixed=0.7, omega_mh2_fixed=0.14157, kde_type='scipy_gaussian',
                                 bandwidth=1, flat=True)

    def test_prior_H0(self):
        logL, bool = CosmoLikelihood.prior_H0(H0=10, H0_min=50, H0_max=100)
        assert bool is False

    def test_prior_omega_mh2(self):
        prior, bool = self.cosmoL.prior_omega_mh2(h=1, omega_m=2, h_max=2)
        assert bool is False

    def test_prior_omega_m(self):
        penalty, bool = self.cosmoL.prior_omega_m(omega_m=0, omega_m_min=0.05, omega_m_max=1)
        assert bool is False

    def test_call(self):
        self.cosmoL.sampling_option = 'H0_only'
        a = [70]
        logL, _ = self.cosmoL(a)
        npt.assert_almost_equal(logL, -11, decimal=-1)

        self.cosmoL.sampling_option = 'H0_omega_m'
        a = [70, 0.3]
        logL, _ = self.cosmoL(a)
        npt.assert_almost_equal(logL, -11, decimal=-1)

        self.cosmoL.sampling_option = "fix_omega_mh2"
        a = [70]
        logL, _ = self.cosmoL(a)
        npt.assert_almost_equal(logL, -11, decimal=-1)

        self.cosmoL.sampling_option = 'H0_omega_m_omega_de'
        a = [70, 0.3, 0.7]
        logL, _ = self.cosmoL(a)
        npt.assert_almost_equal(logL, -11, decimal=-1)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        self.sigma_Dd = 100
        self.sigma_Ddt = 100
        num_samples = 100
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        z_L = 0.3
        z_S = 2
        lensCosmo = LensCosmo(z_L, z_S, cosmo=cosmo)
        self.Dd_true = lensCosmo.D_d
        self.D_dt_true = lensCosmo.D_dt
        D_dt_samples = np.random.normal(self.D_dt_true, self.sigma_Ddt, num_samples)
        D_d_samples = np.random.normal(self.Dd_true, self.sigma_Dd, num_samples)
        self.cosmoL = CosmoLikelihood(z_L, z_S, D_d_samples, D_dt_samples, sampling_option="H0_only", omega_m_fixed=0.3,
                                      omega_lambda_fixed=0.7, omega_mh2_fixed=0.14157, kde_type='scipy_gaussian',
                                      bandwidth=1, flat=True)

        self.cosmoL.sampling_option = 'WRONG'
        with self.assertRaises(ValueError):
            self.cosmoL(a=[])
        with self.assertRaises(ValueError):
            self.cosmoL.likelihood(a=[])
        with self.assertRaises(ValueError):
            self.cosmoL.computeLikelihood(ctx=[])
        with self.assertRaises(ValueError):
            param = CosmoParam(sampling_option='WRONG')
            param.numParam
        with self.assertRaises(ValueError):
            param = CosmoParam(sampling_option='WRONG')
            param.param_bounds




if __name__ == '__main__':
    pytest.main()
