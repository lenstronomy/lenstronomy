__author__ = 'sibirrer'

from scipy import stats
import emcee
import numpy as np
import time
import tempfile
import os
import shutil

from cosmoHammer.util.InMemoryStorageUtil import InMemoryStorageUtil
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import CosmoHammerSampler
from cosmoHammer import MpiCosmoHammerSampler
from lenstronomy.Cosmo.lens_cosmo import FlatLCDM


class CosmoLikelihood(object):
    """
    this class contains the likelihood function of the Strong lensing analysis
    """

    def __init__(self, z_d, z_s, D_d_sample, D_delta_t_sample, sampling_option="H0_only", omega_m_fixed=0.3,
                 omega_mh2_fixed=0.14157, kde_type='scipy_gaussian', bandwidth=1):
        """
        initializes all the classes needed for the chain (i.e. redshifts of lens and source)
        """
        self.z_d = z_d
        self.z_s = z_s
        self.cosmoProp = FlatLCDM(z_lens=z_d, z_source=z_s)
        self._kde_likelihood = KDELikelihood(D_d_sample, D_delta_t_sample, kde_type=kde_type, bandwidth=bandwidth)
        self.sampling_option = sampling_option
        self.omega_m_fixed = omega_m_fixed
        self.omega_mh2_fixed = omega_mh2_fixed

    def X2_chain_H0(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #extract parameters
        H0 = args[0]
        omega_m = self.omega_m_fixed
        logL, bool = self.prior_H0(H0)
        if bool is True:
            logL += self.LCDM_lensLikelihood(H0, omega_m)
            return logL, None
        else:
            pass
        return logL, None

    def X2_chain_omega_mh2(self, args):
        """
        routine to compute the log likelihood given a omega_m h**2 prior fixed
        :param args:
        :return:
        """
        H0 = args[0]
        h = H0/100.
        omega_m = self.omega_mh2_fixed / h**2
        logL, bool = self.prior_omega_mh2(h, omega_m)
        if bool is True:
            logL += self.LCDM_lensLikelihood(H0, omega_m)
            return logL, None
        else:
            pass
        return logL, None

    def X2_chain_H0_omgega_m(self, args):
        """
        routine to compute X^2 value of lens light profile
        :param args:
        :return:
        """
        #extract parameters
        [H0, omega_m] = args
        logL_H0, bool_H0 = self.prior_H0(H0)
        logL_omega_m, bool_omega_m = self.prior_omega_m(omega_m)
        if bool_H0 is True and bool_omega_m is True:
            logL = self.LCDM_lensLikelihood(H0, omega_m)
            return logL + logL_H0 + logL_omega_m, None
        else:
            pass
        return logL_H0 + logL_omega_m, None

    def LCDM_lensLikelihood(self, H0, omega_m):
        Dd = self.cosmoProp.D_d(H0, omega_m)
        Ddt = self.cosmoProp.D_dt(H0, omega_m)
        return self.lensLikelihood(Dd, Ddt)

    def lensLikelihood(self, Dd, Ddt):
        """

        :param alpha:
        :param beta:
        :param sigma_D:
        :return:
        """
        logL = self._kde_likelihood.logLikelihood(Dd, Ddt)
        return logL

    def prior_H0(self, H0, H0_min=0, H0_max=200):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        if H0 < H0_min or H0 > H0_max:
            penalty = -10**15
            return penalty, False
        else:
            return 0, True

    def prior_omega_m(self, omega_m, omega_m_min=0, omega_m_max=1):
        """
        checks whether the parameter omega_m is within the given bounds
        :param omega_m:
        :param omega_m_min:
        :param omega_m_max:
        :return:
        """
        if omega_m < omega_m_min or omega_m > omega_m_max:
            penalty = -10**15
            return penalty, False
        else:
            return 0, True

    def prior_omega_mh2(self, h, omega_m, h_max=2):
        """

        """
        if omega_m > 1 or h > h_max:
            penalty = -10**15
            return penalty, False
        else:
            prior = np.log(np.sqrt(1 + 4*self.omega_mh2_fixed**2/h**6))
            return prior, True

    def __call__(self, a):
        if self.sampling_option == 'H0_only':
            return self.X2_chain_H0(a)
        elif self.sampling_option == 'H0_omega_m':
            return self.X2_chain_H0_omgega_m(a)
        elif self.sampling_option == "fix_omega_mh2":
            return self.X2_chain_omega_mh2(a)
        else:
            raise ValueError("sampling method %s not supported!" % self.sampling_option)

    def likelihood(self, a):
        if self.sampling_option == 'H0_only':
            return self.X2_chain_H0(a)
        elif self.sampling_option == 'H0_omega_m':
            return self.X2_chain_H0_omgega_m(a)
        elif self.sampling_option == "fix_omega_mh2":
            return self.X2_chain_omega_mh2(a)
        else:
            raise ValueError("sampling method %s not supported!" % self.sampling_option)

    def computeLikelihood(self, ctx):
        if self.sampling_option == 'H0_only':
            likelihood, _ = self.X2_chain_H0(ctx.getParams())
        elif self.sampling_option == 'H0_omega_m':
            likelihood, _ = self.X2_chain_H0_omgega_m(ctx.getParams())
        elif self.sampling_option == "fix_omega_mh2":
            likelihood, _ = self.X2_chain_omega_mh2(ctx.getParams())
        else:
            raise ValueError("wrong sampling option specified")
        return likelihood

    def setup(self):
        pass


class CosmoParam(object):
    """
    class for managing the parameters involved
    """
    def __init__(self, sampling_option):
        self.sampling_option = sampling_option

    @property
    def numParam(self):
        if self.sampling_option == "H0_only":
            return 1
        elif self.sampling_option == "H0_omega_m":
            return 2
        else:
            raise ValueError("wrong sampling option specified")

    def param_bounds(self):
        if self.sampling_option == "H0_only" or self.sampling_option == "fix_omega_mh2":
            lowerlimit = [0]
            upperlimit = [200]
        elif self.sampling_option == "H0_omega_m":
            lowerlimit = [0, 0]  # H0, omega_m
            upperlimit = [200, 1]
        else:
            raise ValueError("wrong sampling option specified")
        return lowerlimit, upperlimit


class MCMC_sampler(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, z_d, z_s, D_d_sample, D_dt_sample, sampling_option="H0_only", omega_m_fixed=0.3,
                 omega_mh2_fixed=0.14157, kde_type='scipy_gaussian', bandwidth=1):
        """
        initialise the classes of the chain and for parameter options
        """
        self.chain = CosmoLikelihood(z_d, z_s, D_d_sample, D_dt_sample, sampling_option=sampling_option,
                                     omega_m_fixed=omega_m_fixed, omega_mh2_fixed=omega_mh2_fixed,
                                     kde_type=kde_type, bandwidth=bandwidth)
        self.cosmoParam = CosmoParam(sampling_option)

    def mcmc_emcee(self, n_walkers, n_run, n_burn, mean_start, sigma_start):
        """
        returns the mcmc analysis of the parameter space
        """
        sampler = emcee.EnsembleSampler(n_walkers, self.cosmoParam.numParam, self.chain.likelihood)
        p0 = emcee.utils.sample_ball(mean_start, sigma_start, n_walkers)
        new_pos, _, _, _ = sampler.run_mcmc(p0, n_burn)
        sampler.reset()
        store = InMemoryStorageUtil()
        for pos, prob, _, _ in sampler.sample(new_pos, iterations=n_run):
            store.persistSamplingValues(pos, prob, None)
        return store.samples

    def mcmc_CH(self, walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi_monch=False):
        """
        runs mcmc on the parameter space given parameter bounds with CosmoHammerSampler
        returns the chain
        """
        lowerLimit, upperLimit = self.cosmoParam.param_bounds()
        params = np.array([mean_start, lowerLimit, upperLimit, sigma_start]).T


        chain = LikelihoodComputationChain(
            min=lowerLimit,
            max=upperLimit)

        temp_dir = tempfile.mkdtemp("Hammer")
        file_prefix = os.path.join(temp_dir, "logs")

        # chain.addCoreModule(CambCoreModule())
        chain.addLikelihoodModule(self.chain)
        chain.setup()

        store = InMemoryStorageUtil()
        if mpi_monch is True:
            sampler = MpiCosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=file_prefix,
            walkersRatio=walkerRatio,
            burninIterations=n_burn,
            sampleIterations=n_run,
            threadCount=1,
            initPositionGenerator=init_pos,
            storageUtil=store)
        else:
            sampler = CosmoHammerSampler(
                params=params,
                likelihoodComputationChain=chain,
                filePrefix=file_prefix,
                walkersRatio=walkerRatio,
                burninIterations=n_burn,
                sampleIterations=n_run,
                threadCount=threadCount,
                initPositionGenerator=init_pos,
                storageUtil=store)
        time_start = time.time()
        if sampler.isMaster():
            print('Computing the MCMC...')
            print('Number of walkers = ', len(mean_start)*walkerRatio)
            print('Burn-in itterations: ', n_burn)
            print('Sampling itterations:', n_run)
        sampler.startSampling()
        if sampler.isMaster():
            time_end = time.time()
            print(time_end - time_start, 'time taken for MCMC sampling')
        # if sampler._sampler.pool is not None:
        #     sampler._sampler.pool.close()
        try:
            shutil.rmtree(temp_dir)
        except Exception as ex:
            print(ex)
            pass
        return store.samples


class KDELikelihood(object):
    """
    class that samples the cosmographic likelihood given a distribution of points in the 2-dimensional distribution
    of D_d and D_delta_t
    """
    def __init__(self, D_d_sample, D_delta_t_sample, kde_type='scipy_gaussian', bandwidth=1):
        """

        :param D_d_sample: 1-d numpy array of angular diamter distances to the lens plane
        :param D_delta_t_sample: 1-d numpy array of time-delay distances
        """
        values = np.vstack([D_d_sample, D_delta_t_sample])
        if kde_type == 'scipy_gaussian':
            self._PDF_kernel = stats.gaussian_kde(values)
        else:
            from sklearn.neighbors import KernelDensity
            self._kde = KernelDensity(bandwidth=bandwidth, kernel=kde_type)
            values = np.vstack([D_d_sample, D_delta_t_sample])
            self._kde.fit(values.T)
        self._kde_type = kde_type



    def logLikelihood(self, D_d, D_delta_t):
        """
        likelihood of the data (represented in the distribution of this class) given a model with predicted angular
        diameter distances.

        :param D_d: model predicted angular diameter distance
        :param D_delta_t: model predicted time-delay distance
        :return: loglikelihood (log of KDE value)
        """
        if self._kde_type == 'scipy_gaussian':
            density = self._PDF_kernel([D_d, D_delta_t])
            logL = np.log(density)
        else:
            x = np.array([[D_d], [D_delta_t]])
            logL = self._kde.score_samples(x.T)
        return logL
