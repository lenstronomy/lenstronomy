__author__ = 'sibirrer'

import numpy as np
from astrofunc.util import Util_class

from lenstronomy.Cosmo.time_delay_sampling import TimeDelaySampling
from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.MCMC.compare import Compare
from lenstronomy.Workflow.parameters import Param


class MCMC_chain(object):
    """
    this class contains the routines to run a MCMC process with one single image
    """
    def __init__(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else):
        """
        initializes all the classes needed for the chain
        """
        # print('initialized on cpu', threading.current_thread())
        self.util_class = Util_class()

        self.sampling_option = kwargs_options.get('X2_type', 'image')
        self.makeImage = MakeImage(kwargs_options, kwargs_data, kwargs_psf)
        self.param = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else)
        self.compare = Compare(kwargs_options)
        self.lowerLimit, self.upperLimit = self.param.param_bounds()
        self.timeDelay = TimeDelaySampling()
        self.time_delay = kwargs_options.get('time_delay', False)
        if self.time_delay is True:
            self.delays_measured = kwargs_data['time_delays']
            self.delays_errors = kwargs_data['time_delays_errors']
        self.inv_bool = kwargs_options.get('source_marg', False)  # whether to fully invert the covariance matrix for marginalization
        self.priors_bool = kwargs_options.get('priors', False)
        if self.priors_bool:
            self.kwargs_priors = kwargs_options['kwargs_priors']

    def X2_chain_image(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        im_sim, model_error, cov_matrix, param = self.makeImage.image_linear_solve(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, inv_bool=self.inv_bool)
        #compute X^2
        X = self.makeImage.Data.reduced_residuals(im_sim, model_error)
        logL = self.compare.get_log_likelihood(X, cov_matrix=cov_matrix)
        logL -= self.check_bounds(args, self.lowerLimit, self.upperLimit)
        # logL -= self.bounds_convergence(kwargs_lens)
        if self.time_delay is True:
            logL += self.logL_delay(kwargs_lens, kwargs_source, kwargs_else)
        if self.priors_bool:
            logL += self.priors(kwargs_lens, self.kwargs_priors)
        return logL, None

    def X2_chain_catalogue(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chain
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        x_mapped, y_mapped = self.makeImage.LensModel.ray_shooting(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_lens, kwargs_else)
        #compute X^2
        X2 = self.compare.compare_distance(x_mapped, y_mapped)*1000
        X2 += self.check_bounds(args, self.lowerLimit, self.upperLimit)
        if self.priors_bool:
            X2 -= self.priors(kwargs_lens, self.kwargs_priors)
        return -X2, None

    def priors(self, kwargs_lens, kwargs_priors):
        """

        :param kwargs_lens:
        :param kwargs_priors:
        :return:
        """
        prior = 0
        if 'gamma_ext' in kwargs_lens and 'gamma_ext' in kwargs_priors and 'gamma_ext_sigma' in kwargs_priors:
            prior -= (kwargs_lens['gamma_ext']-kwargs_priors['gamma_ext'])**2/(2*kwargs_priors['gamma_ext_sigma'])**2
        if 'psi_ext' in kwargs_lens and 'psi_ext' in kwargs_priors and 'psi_ext_sigma' in kwargs_priors:
            prior -= (kwargs_lens['psi_ext']-kwargs_priors['psi_ext'])**2/(2*kwargs_priors['psi_ext_sigma'])**2
        return prior

    def check_bounds(self, args, lowerLimit, upperLimit):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        penalty = 0
        for i in range(0, len(args)):
            if args[i] < lowerLimit[i] or args[i] > upperLimit[i]:
                penalty = 10**15#np.NaN #10**10
                #print(i, args[i], lowerLimit[i], upperLimit[i])
                #print("warning!!!")
        return penalty

    def bounds_convergence(self, kwargs_lens, kwargs_else=None):
        """
        bounds computed from kwargs
        """
        convergence = self.makeImage.LensModel.kappa(kwargs_lens, kwargs_else=kwargs_else)
        if np.min(np.array(convergence)) < -0.1:
            return 10**10
        else:
            return 0

    def logL_delay(self, kwargs_lens, kwargs_source, kwargs_else):
        """
        routine to compute the log likelihood of the time delay distance
        :param args:
        :return:
        """
        delay_arcsec = self.makeImage.fermat_potential(kwargs_lens, kwargs_else)
        D_dt_model = kwargs_else['delay_dist']
        delay_days = self.timeDelay.days_D_model(delay_arcsec, D_dt_model)
        logL = self.compare.delays(delay_days, self.delays_measured, self.delays_errors)
        return logL

    def __call__(self, a):
        if self.sampling_option == 'image':
            return self.X2_chain_image(a)
        elif self.sampling_option == 'catalogue':
            return self.X2_chain_catalogue(a)
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)

    def likelihood(self, a):
        if self.sampling_option == 'image':
            return self.X2_chain_image(a)
        elif self.sampling_option == 'catalogue':
            return self.X2_chain_catalogue(a)
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)

    def computeLikelihood(self, ctx):
        if self.sampling_option == 'image':
            likelihood, _ = self.X2_chain_image(ctx.getParams())
        elif self.sampling_option == 'catalogue':
            likelihood, _ = self.X2_chain_catalogue(ctx.getParams())
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)
        return likelihood

    def setup(self):
        pass

    def numData_points(self):
        """
        returns the effective number of data points considered in the X2 estimation to compute the reduced X2 value
        """
        if type(self.makeImage.Data.mask) == int:
            n = self.makeImage.Data._nx * self.makeImage.Data._ny
        else:
            n = np.sum(self.makeImage.Data.mask)
        num_param, _ = self.param.num_param()
        return n - num_param - 1
