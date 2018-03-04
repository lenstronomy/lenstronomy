__author__ = 'sibirrer'

import numpy as np
from lenstronomy.Workflow.parameters import Param
import lenstronomy.Util.util as util
import lenstronomy.Util.constants as const
import lenstronomy.Util.class_creator as class_creator


class LikelihoodModule(object):
    """
    this class contains the routines to run a MCMC process with one single image
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_fixed, kwargs_lower, kwargs_upper, kwargs_lens_init=None, compute_bool=None):
        """
        initializes all the classes needed for the chain
        """
        # print('initialized on cpu', threading.current_thread())
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps = kwargs_fixed
        self.Multiband = class_creator.creat_multiband(multi_band_list, kwargs_model)
        self.lensModel = self.Multiband.lensModel
        self.param = Param(kwargs_model, kwargs_constraints, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_lens_init=kwargs_lens_init)
        kwargs_lens_lower, kwargs_source_lower, kwargs_lens_light_lower, kwargs_else_lower = kwargs_lower
        kwargs_lens_upper, kwargs_source_upper, kwargs_lens_light_upper, kwargs_else_upper = kwargs_upper
        self.lower_limit = self.param.setParams(kwargs_lens_lower, kwargs_source_lower, kwargs_lens_light_lower, kwargs_else_lower, bounds='lower')
        self.upper_limit = self.param.setParams(kwargs_lens_upper, kwargs_source_upper, kwargs_lens_light_upper,
                                           kwargs_else_upper, bounds='upper')

        self._check_bounds = kwargs_likelihood.get('check_bounds', True)
        self._point_source_likelihood = kwargs_likelihood.get('point_source_likelihood', False)
        self._position_sigma = kwargs_likelihood.get('position_uncertainty', 0.004)
        self._image_likelihood = kwargs_likelihood.get('image_likelihood', True)
        self._check_solver = kwargs_likelihood.get('check_solver', False)
        self._solver_tolerance = kwargs_likelihood.get('solver_tolerance', 0.001)
        self._force_no_add_image = kwargs_likelihood.get('force_no_add_image', False)
        self._source_marg = kwargs_likelihood.get('source_marg',
                                                  False)  # whether to fully invert the covariance matrix for marginalization
        self._num_bands = len(multi_band_list)
        if compute_bool is None:
            self._compute_bool = [True] * self._num_bands
        else:
            if not len(compute_bool) == self._num_bands:
                raise ValueError('compute_bool statement has not the same range as number of bands available!')
            self._compute_bool = compute_bool

        # this part is not yet fully implemented
        self.time_delay = kwargs_likelihood.get('time_delay', False)
        if self.time_delay is True:
            self.delays_measured = kwargs_likelihood['time_delays']
            self.delays_errors = kwargs_likelihood['time_delays_errors']
            raise ValueError("cosmological sampling of time-delays not implemented in this version of lenstronomy yet!"
                             " Please get in touch with the the developers.")
        self.priors_bool = kwargs_likelihood.get('priors', False)
        if self.priors_bool:
            self._prior_module = kwargs_likelihood['prior_module']

    def X2_chain(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = self.param.getParams(args)
        #generate image and computes likelihood
        self.Multiband.reset_point_source_cache()
        logL = 0
        if self._check_bounds:
            penalty, bound_hit = self.check_bounds(args, self.lower_limit, self.upper_limit)
            logL -= penalty
            if bound_hit:
                return logL, None
        if self._image_likelihood:
            logL += self.Multiband.likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                                          source_marg=self._source_marg, compute_bool=self._compute_bool)
            x_mins, y_mins = self.Multiband.image_positions(kwargs_ps, kwargs_lens)
        if self._point_source_likelihood:
            logL += self.likelihood_image_pos(kwargs_lens, kwargs_ps, self._position_sigma)
        # logL -= self.bounds_convergence(kwargs_lens)
        if self.time_delay is True:
            logL += self.logL_delay(kwargs_lens, kwargs_ps)
        if self.priors_bool:
            logL += self.prior_compute(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        if self._check_solver is True:
            logL -= self.check_solver(kwargs_lens, kwargs_ps, self._solver_tolerance)
        if self._force_no_add_image:
            bool = self.check_additional_images(kwargs_ps, kwargs_lens)
            if bool:
                logL -= 10**10
        return logL, None

    def check_solver(self, kwargs_lens, kwargs_ps, tolerance):
        """
        test whether the image positions map back to the same source position
        :param kwargs_lens:
        :param kwargs_ps:
        :return:
        """
        # TODO: make this part of the PointSource class
        if 'ra_image' in kwargs_ps[0]:
            ra_image, dec_image = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
        else:
            ra_image_list, dec_image_list = self.Multiband.image_positions(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
            ra_image, dec_image = ra_image_list[0], dec_image_list[0]
        source_x, source_y = self.lensModel.ray_shooting(ra_image, dec_image,
                                                             kwargs_lens)
        dist = np.sqrt((source_x - source_x[0])**2 + (source_y - source_y[0])**2)
        if np.max(dist) > tolerance:
            return np.sum(dist) * 10**10
        return 0

    def check_additional_images(self, kwargs_ps, kwargs_lens):
        """
        checks whether additional images have been found and placed in kwargs_else
        :param kwargs_else:
        :return:
        """
        # TODO: make this part of the PointSource class
        ra_image_list, dec_image_list = self.Multiband.image_positions(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        if len(ra_image_list[0]) > self.param.num_point_source_images:
            return True
        else:
            return False

    def prior_compute(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        if not hasattr(self._prior_module, 'likelihood'):
            raise ValueError("prior module instance needs a definition 'likelihood")
        logL = self._prior_module.likelihood(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        return logL

    def likelihood_image_pos(self, kwargs_lens, kwargs_ps, sigma):
        """

        :param x_lens_model: image position of lens model
        :param y_lens_model: image position of lens model
        :param x_image: image position of image data
        :param y_image: image position of image data
        :param sigma: likelihood sigma
        :return: log likelihood of model given image positions
        """
        # TODO: make this part of the PointSource class
        x_image = kwargs_ps[0]['ra_image']
        y_image = kwargs_ps[0]['dec_image']
        ra_image_list, dec_image_list = self.Multiband.image_positions(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        num_image = len(ra_image_list[0])
        if num_image != len(x_image):
            return -10**15
        dist = util.min_square_dist(ra_image_list[0], dec_image_list[0], x_image, y_image)
        logL = - np.sum(dist/sigma**2)/2
        return logL

    def check_bounds(self, args, lowerLimit, upperLimit):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        penalty = 0
        bound_hit = False
        for i in range(0, len(args)):
            if args[i] < lowerLimit[i] or args[i] > upperLimit[i]:
                penalty = 10**15
                bound_hit = True
        return penalty, bound_hit

    def logL_delay(self, kwargs_lens, kwargs_cosmo):
        """
        routine to compute the log likelihood of the time delay distance
        :param args:
        :return:
        """
        delay_arcsec = self.Multiband.fermat_potential(kwargs_lens, kwargs_cosmo)
        D_dt_model = kwargs_cosmo['delay_dist']
        delay_days = const.delay_arcsec2days(delay_arcsec, D_dt_model)
        logL = self._logL_delays(delay_days, self.delays_measured, self.delays_errors)
        return logL

    def _logL_delays(self, delays_model, delays_measured, delays_errors):
        """
        log likelihoood of modeled delays vs measured time delays under considerations of errors

        :param delays_model: n delays of the model (not relative delays)
        :param delays_measured: relative delays (1-2,1-3,1-4) relative to the first in the list
        :param delays_errors: gaussian errors on the measured delays
        :return: log likelihood of data given model
        """
        delta_t_model = np.array(delays_model[1:]) - delays_model[0]
        logL = np.sum(-(delta_t_model - delays_measured) ** 2 / (2 * delays_errors ** 2))
        return logL

    def effectiv_numData_points(self):
        """
        returns the effective number of data points considered in the X2 estimation to compute the reduced X2 value
        """
        n = self.Multiband.numData_evaluate(compute_bool=self._compute_bool)
        num_param, _ = self.param.num_param()
        return n - num_param - 1

    def __call__(self, a):
        return self.X2_chain(a)

    def likelihood(self, a):
        return self.X2_chain(a)

    def computeLikelihood(self, ctx):
        logL, _ = self.X2_chain(ctx.getParams())
        return logL

    def setup(self):
        pass
