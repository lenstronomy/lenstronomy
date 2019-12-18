__author__ = 'sibirrer'

import emcee
import numpy as np
from lenstronomy.Cosmo.lcdm import LCDM
from lenstronomy.Cosmo.kde_likelihood import KDELikelihood


class LensLikelihood(object):
    """
    class for evaluating single lens likelihood
    """
    def __init__(self, z_d, z_s, D_d_sample, D_delta_t_sample, kde_type='scipy_gaussian', bandwidth=1, flat=True):
        """

        :param z_d: lens redshift
        :param z_s: source redshift
        :param D_d_sample: angular diameter to the lens posteriors (in physical Mpc)
        :param D_delta_t_sample: time-delay distance posteriors (in physical Mpc)
        :param kde_type: kernel density estimator type (see KDELikelihood class)
        :param bandwidth: width of kernel (in same units as the angular diameter quantities)
        :param flat: boolean, flat or curved cosmology
        """
        self._z_d = z_d
        self._z_s = z_s
        self._cosmoProp = LCDM(z_lens=z_d, z_source=z_s, flat=flat)
        self._kde_likelihood = KDELikelihood(D_d_sample, D_delta_t_sample, kde_type=kde_type, bandwidth=bandwidth)

    def lens_log_likelihood(self, H0, omega_m, Ode0=None):
        Dd = self._cosmoProp.D_d(H0, omega_m, Ode0)
        Ddt = self._cosmoProp.D_dt(H0, omega_m, Ode0)
        return self._kde_likelihood.logLikelihood(Dd, Ddt)


class LensSampleLikelihood(object):
    """
    class to evaluate the likelihood of a cosmology given a sample of angular diameter posteriors
    Currently this class does not include possible covariances between the lens samples
    """
    def __init__(self, kwargs_lens_list, flat=True):
        """

        :param kwargs_lens_list: keyword argument list specifying the arguments of the LensLikelihood class
        :param flat: boolean, flat or curved cosmology
        """
        self._lens_list = []
        for kwargs_lens in kwargs_lens_list:
            self._lens_list.append(LensLikelihood(flat=flat, **kwargs_lens))

    def log_likelihood(self, H0, omega_m, Ode0=None):
        """

        :param H0: Hubble constant in km/s/Mpc
        :param omega_m: Omega_m
        :param Ode0: dark energy density
        :return: log likelihood of the combined lenses
        """
        logL = 0
        for lens in self._lens_list:
            logL += lens.lens_log_likelihood(H0, omega_m, Ode0)
        return logL


class CosmoLikelihood(object):
    """
    this class contains the likelihood function of the Strong lensing analysis
    """

    def __init__(self, kwargs_lens_list, sampling_option="H0_only", omega_m_fixed=0.3,
                 omega_lambda_fixed=0.7, omega_mh2_fixed=0.14157, flat=True):
        """

        :param kwargs_lens_list: keyword argument list specifying the arguments of the LensLikelihood class
        :param sampling_option: string indicating what cosmology to sample from, supported are: 'H0_only', 'H0_omega_m',
         "fix_omega_mh2", 'H0_omega_m_omega_de'
        :param omega_m_fixed: float, value to be fixed if Omega_m is kept fixed
        :param omega_lambda_fixed: float, value to be fixed if Omega_lambda is kept fixed
        :param omega_mh2_fixed: float, value to be fixed if Omega_m h**2 is held fixed
        :param flat: boolean, flat or curved cosmology
        """

        self._likelihoodLensSample = LensSampleLikelihood(kwargs_lens_list, flat=flat)
        self.sampling_option = sampling_option
        self.omega_m_fixed = omega_m_fixed
        self.omega_mh2_fixed = omega_mh2_fixed
        self._omega_lambda_fixed = omega_lambda_fixed

    def lcdm_likelihood(self, H0, omega_m, Ode0):
        """

        :param H0:
        :param omega_m:
        :param Ode0:
        :return:
        """
        return self._likelihoodLensSample.log_likelihood(H0, omega_m, Ode0)

    def X2_chain_H0(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chain
        """
        #extract parameters
        H0 = args[0]
        omega_m = self.omega_m_fixed
        Ode0 = self._omega_lambda_fixed
        logL, bool = self.prior_H0(H0)
        if bool is True:
            logL += self.lcdm_likelihood(H0, omega_m, Ode0)
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
        Ode0 = self._omega_lambda_fixed
        logL, bool = self.prior_omega_mh2(h, omega_m)
        if bool is True:
            logL += self.lcdm_likelihood(H0, omega_m, Ode0)
        return logL, None

    def X2_chain_H0_omgega_m(self, args):
        """
        routine to compute X^2
        :param args:
        :return:
        """
        #extract parameters
        [H0, omega_m] = args
        Ode0 = self._omega_lambda_fixed
        logL_H0, bool_H0 = self.prior_H0(H0)
        logL_omega_m, bool_omega_m = self.prior_omega_m(omega_m)
        logL = logL_H0 + logL_omega_m
        if bool_H0 is True and bool_omega_m is True:
            logL += self.lcdm_likelihood(H0, omega_m, Ode0)
        return logL + logL_H0 + logL_omega_m, None

    def X2_chain_H0_omgega_m_omega_de(self, args):
        """
        routine to compute X^2
        :param args:
        :return:
        """
        #extract parameters
        [H0, omega_m, Ode0] = args
        logL_H0, bool_H0 = self.prior_H0(H0)
        logL_omega_m, bool_omega_m = self.prior_omega_m(omega_m)
        logL = logL_H0 + logL_omega_m
        if bool_H0 is True and bool_omega_m is True:
            logL += self.lcdm_likelihood(H0, omega_m, Ode0)
        return logL + logL_H0 + logL_omega_m, None

    @staticmethod
    def prior_H0(H0, H0_min=0, H0_max=200):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        if H0 < H0_min or H0 > H0_max:
            penalty = -np.inf
            return penalty, False
        else:
            return 0, True

    @staticmethod
    def prior_omega_m(omega_m, omega_m_min=0, omega_m_max=1):
        """
        checks whether the parameter omega_m is within the given bounds
        :param omega_m:
        :param omega_m_min:
        :param omega_m_max:
        :return:
        """
        if omega_m < omega_m_min or omega_m > omega_m_max:
            penalty = -np.inf
            return penalty, False
        else:
            return 0, True

    def prior_omega_mh2(self, h, omega_m, h_max=2):
        """

        """
        if omega_m > 1 or h > h_max:
            penalty = -np.inf
            return penalty, False
        else:
            prior = np.log(np.sqrt(1 + 4*self.omega_mh2_fixed**2/h**6))
            return prior, True

    def likelihood(self, a):
        logL, _ = self._likelihood(a)
        return logL

    def _likelihood(self, a):
        if self.sampling_option == 'H0_only':
            return self.X2_chain_H0(a)
        elif self.sampling_option == 'H0_omega_m':
            return self.X2_chain_H0_omgega_m(a)
        elif self.sampling_option == "fix_omega_mh2":
            return self.X2_chain_omega_mh2(a)
        elif self.sampling_option == 'H0_omega_m_omega_de':
            return self.X2_chain_H0_omgega_m_omega_de(a)
        else:
            raise ValueError("sampling method %s not supported!" % self.sampling_option)


class CosmoParam(object):
    """
    class for managing the parameters involved
    """
    def __init__(self, sampling_option, lower_limit=[0, 0, 0], upper_limit=[200, 1, 1]):
        """

        :param sampling_option: string, sampling option
        :param lower_limit: lower limit, array [H0, Omega_m, Omega_dm]
        :param upper_limit: upper limit, array [H0, Omega_m, Omega_dm]
        """
        self.sampling_option = sampling_option
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit

    @property
    def numParam(self):
        if self.sampling_option == "H0_only" or self.sampling_option == "fix_omega_mh2":
            return 1
        elif self.sampling_option == "H0_omega_m":
            return 2
        elif self.sampling_option == 'H0_omega_m_omega_de':
            return 3
        else:
            raise ValueError("wrong sampling option specified")

    @property
    def param_bounds(self):
        if self.sampling_option == "H0_only" or self.sampling_option == "fix_omega_mh2":
            lowerlimit = [self._lower_limit[0]]
            upperlimit = [self._upper_limit[0]]
        elif self.sampling_option == "H0_omega_m":
            lowerlimit = self._lower_limit[0:2] # H0, omega_m
            upperlimit = self._upper_limit[0:2]
        elif self.sampling_option == 'H0_omega_m_omega_de':
            lowerlimit = self._lower_limit[0:3]
            upperlimit = self._upper_limit[0:3]
        else:
            raise ValueError("wrong sampling option specified")
        return lowerlimit, upperlimit


class MCMCSampler(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, kwargs_lens_list, sampling_option="H0_only", omega_m_fixed=0.3, omega_mh2_fixed=0.14157,
                 flat=True, lower_limit=[0, 0, 0], upper_limit=[200, 1, 1]):
        """
        initialise the classes of the chain and for parameter options
        """
        self.chain = CosmoLikelihood(kwargs_lens_list, sampling_option=sampling_option, omega_m_fixed=omega_m_fixed,
                                     omega_mh2_fixed=omega_mh2_fixed, flat=flat)
        self.cosmoParam = CosmoParam(sampling_option, lower_limit=lower_limit, upper_limit=upper_limit)

    def mcmc_emcee(self, n_walkers, n_run, n_burn, mean_start, sigma_start):
        """
        returns the mcmc analysis of the parameter space
        """
        sampler = emcee.EnsembleSampler(n_walkers, self.cosmoParam.numParam, self.chain.likelihood, args=())
        p0 = emcee.utils.sample_ball(mean_start, sigma_start, n_walkers)
        #p0 = mean_start *np.random.randn(n_walkers, self.cosmoParam.numParam)
        sampler.run_mcmc(p0, n_burn+n_run, progress=True)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        return flat_samples
