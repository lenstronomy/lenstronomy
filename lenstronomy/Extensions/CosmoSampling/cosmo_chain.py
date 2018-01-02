__author__ = 'sibirrer'

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
from lenstronomy.Cosmo.cosmo_properties_pycosmo import CosmoProp
from lenstronomy.FunctionSet.porb_density import SkewGaussian


class CosmoLikelihood(object):
    """
    this class contains the likelihood function of the Strong lensing analysis
    """

    def __init__(self, z_d, z_s, alpha, beta, sigma_D, skw=0, sampling_option="H0_only", omega_m_fixed=0.3, omega_mh2_fixed=0.14157, skewness=True, kappa_d=0):
        """
        initializes all the classes needed for the chain (i.e. redshifts of lens and source)
        """
        self.z_d = z_d
        self.z_s = z_s
        self.cosmoProp = CosmoProp(z_lens=z_d, z_source=z_s)
        self.alpha = alpha
        self.beta = beta
        self.sigma_D = sigma_D
        self.source_prior = False
        self.kappa_d = kappa_d
        if skw != 0 and skewness is True:
            if skw < -1 or skw > 1:
                raise ValueError("skewness parameter %s out of allowed range [-1,1]" % skw)
            self.skw = skw
            self.skewGaussian = SkewGaussian()
            self.e_skw, self.w_skw, self.a_skw = self.skewGaussian.map_mu_sigma_skw(0, sigma_D, skw)
            self.skewness = True
        else:
            self.skewness = False

        self.sampling_option = sampling_option
        self.omega_m_fixed = omega_m_fixed
        self.omega_mh2_fixed = omega_mh2_fixed


    def X2_chain_H0(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #extract parameters
        H0 = args
        omega_m = self.omega_m_fixed
        logL, bool = self.prior_H0(H0)
        if bool is True:
            logL += self.LCDM_lensLikelihood(H0, omega_m, self.alpha, self.beta, self.sigma_D)
            return logL, None
        else:
            pass
        return logL, None

    def X2_chain_omega_mh2(self, args):
        """
        routine to compute the log likelihood of given a omega_m h**2 prior fixed
        :param args:
        :return:
        """
        H0 = args
        h = H0/100.
        omega_m = self.omega_mh2_fixed / h**2
        logL, bool = self.prior_omega_mh2(h, omega_m)
        if bool is True:
            logL += self.LCDM_lensLikelihood(H0, omega_m, self.alpha, self.beta, self.sigma_D)
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
            logL = self.LCDM_lensLikelihood(H0, omega_m, self.alpha, self.beta, self.sigma_D)
            return logL + logL_H0 + logL_omega_m, None
        else:
            pass
        return logL_H0 + logL_omega_m, None

    def LCDM_lensLikelihood(self, H0, omega_m, alpha, beta, sigma_D):
        Dd = self.cosmoProp.D_d(H0, omega_m) * (1 - self.kappa_d)
        Ds = self.cosmoProp.D_s(H0, omega_m)
        Dds = self.cosmoProp.D_ds(H0, omega_m)
        return self.lensLikelihood(Dd, Ds, Dds, alpha, beta, sigma_D)

    def lensLikelihood(self, Dd, Ds, Dds, alpha, beta, sigma_D):
        """

        :param alpha:
        :param beta:
        :param sigma_D:
        :return:
        """
        delta_y = Dds/(Dd*Ds)-(alpha*np.log(Dd)+beta)
        if self.skewness is True:
            logL = np.log(self.skewGaussian.pdf(x=delta_y, e=self.e_skw, w=self.w_skw, a=self.a_skw))
        else:
            logL = -delta_y**2/(2*sigma_D**2)
        if self.source_prior:
            logL += self.prior_beta(Dds/(Dd*Ds))
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

    def init_source_prior(self, beta0, alpha_beta, alpha_luminosity):
        """
        initializes the prior description for the source scale
        :param beta0: intercept of ln Dd vs beta relation
        :param alpha_beta: slope of ln Dd vs beta relation
        :param alpha_luminosity: power-law slope of the luminosity function
        :return: initialized parameter in self
        """
        self.beta0 = beta0
        self.alpha_beta = alpha_beta
        self.alpha_luminosity = alpha_luminosity
        self.source_prior = True

    def prior_beta(self, Dds_DdDs):
        beta = self.beta0 + self.alpha_beta*Dds_DdDs
        beta = max(0.01, beta)
        p_beta = beta**(2*self.alpha_luminosity+1)
        return np.log(p_beta)

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
            raise ValueError("wrong sampling option specified")

    def likelihood(self, a):
        if self.sampling_option == 'H0_only':
            return self.X2_chain_H0(a)
        elif self.sampling_option == 'H0_omega_m':
            return self.X2_chain_H0_omgega_m(a)
        elif self.sampling_option == "fix_omega_mh2":
            return self.X2_chain_omega_mh2(a)
        else:
            raise ValueError("wrong sampling option specified")

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
    def __init__(self, z_d, z_s, alpha, beta, sigma_D, skw=0, sampling_option="H0_only", omega_m_fixed=0.3, omega_mh2_fixed=0.14157, skewness=True, kappa_d=0):
        """
        initialise the classes of the chain and for parameter options
        """
        self.chain = CosmoLikelihood(z_d, z_s, alpha, beta, sigma_D, skw=skw, sampling_option=sampling_option, omega_m_fixed=omega_m_fixed, omega_mh2_fixed=omega_mh2_fixed, skewness=skewness, kappa_d=kappa_d)
        self.cosmoParam = CosmoParam(sampling_option)

    def init_source_priors(self, beta0, alpha_beta, alpha_luminosity):
        self.chain.init_source_prior(beta0, alpha_beta, alpha_luminosity)

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

