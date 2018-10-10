__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.constants as const


class LikelihoodModule(object):
    """
    this class contains the routines to run a MCMC process with one single image
    """
    def __init__(self, imSim_class, param_class, kwargs_likelihood):
        """
        initializes all the classes needed for the chain
        """
        self.imSim = imSim_class
        self.lensModel = self.imSim.LensModel
        self.param = param_class
        self._lower_limit, self._upper_limit = self.param.param_limits()
        # this part is not yet fully implemented
        self._time_delay_likelihood = kwargs_likelihood.get('time_delay_likelihood', False)
        if self._time_delay_likelihood is True:
            self._delays_measured = np.array(kwargs_likelihood['time_delays_measured'])
            self._delays_errors = np.array(kwargs_likelihood['time_delays_uncertainties'])

        self._check_bounds = kwargs_likelihood.get('check_bounds', True)
        self._point_source_likelihood = kwargs_likelihood.get('point_source_likelihood', False)
        self._position_sigma = kwargs_likelihood.get('position_uncertainty', 0.004)
        self._image_likelihood = kwargs_likelihood.get('image_likelihood', True)
        self._check_solver = kwargs_likelihood.get('check_solver', False)
        self._check_positive_flux = kwargs_likelihood.get('check_positive_flux', False)
        self._solver_tolerance = kwargs_likelihood.get('solver_tolerance', 0.001)
        self._force_no_add_image = kwargs_likelihood.get('force_no_add_image', False)
        self._source_marg = kwargs_likelihood.get('source_marg', False)  # whether to fully invert the covariance matrix for marginalization
        self._restrict_number_images = kwargs_likelihood.get('restrict_image_number', False)
        self._max_num_images = kwargs_likelihood.get('max_num_images', self.param.num_point_source_images)
        self._num_bands = self.imSim.num_bands
        self._compute_bool = kwargs_likelihood.get('bands_compute', [True] * self._num_bands)
        if not len(self._compute_bool) == self._num_bands:
            raise ValueError('compute_bool statement has not the same range as number of bands available!')

    @property
    def param_limits(self):
        return self._lower_limit, self._upper_limit

    def logL(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo = self.param.getParams(args)
        #generate image and computes likelihood
        self.imSim.reset_point_source_cache()
        logL = 0
        if self._check_bounds is True:
            penalty, bound_hit = self.check_bounds(args, self._lower_limit, self._upper_limit)
            logL -= penalty
            if bound_hit:
                return logL, None
        if self._image_likelihood is True:
            logL += self.imSim.likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                                           source_marg=self._source_marg, compute_bool=self._compute_bool)
        if self._point_source_likelihood is True:
            logL += self.likelihood_image_pos(kwargs_lens, kwargs_ps, self._position_sigma)
        if self._time_delay_likelihood is True:
            logL += self.logL_delay(kwargs_lens, kwargs_ps, kwargs_cosmo)
        if self._check_solver is True:
            logL -= self.solver_penalty(kwargs_lens, kwargs_ps, self._solver_tolerance)
        if self._force_no_add_image:
            bool = self.check_additional_images(kwargs_ps, kwargs_lens)
            if bool is True:
                logL -= 10**10
        if self._restrict_number_images is True:
            ra_image_list, dec_image_list = self.imSim.image_positions(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
            if len(ra_image_list[0]) > self._max_num_images:
                logL -= 10**10
        if self._check_positive_flux is True:
            logL -= self.check_positive_flux(kwargs_source, kwargs_lens_light, kwargs_ps)
        return logL, None

    def solver_penalty(self, kwargs_lens, kwargs_ps, tolerance):
        """
        test whether the image positions map back to the same source position
        :param kwargs_lens:
        :param kwargs_ps:
        :return: add penalty when solver does not find a solution
        """
        dist = self.param.check_solver(kwargs_lens, kwargs_ps)
        if dist > tolerance:
            return dist * 10**10
        return 0

    def check_additional_images(self, kwargs_ps, kwargs_lens):
        """
        checks whether additional images have been found and placed in kwargs_ps
        :param kwargs_ps: point source kwargs
        :return: bool, True if more image positions are found than originally been assigned
        """
        ra_image_list, dec_image_list = self.imSim.image_positions(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        if len(ra_image_list) > 0:
            if len(ra_image_list[0]) > self.param.num_point_source_images:
                return True
        else:
            return False

    def likelihood_image_pos(self, kwargs_lens, kwargs_ps, sigma):
        """

        :param x_lens_model: image position of lens model
        :param y_lens_model: image position of lens model
        :param x_image: image position of image data
        :param y_image: image position of image data
        :param sigma: likelihood sigma
        :return: log likelihood of model given image positions
        """
        # TODO think of where to put it, it used specific keyword arguments
        # TODO does this work with source position defined point source and required arguemnts 'ra_image'?
        if not 'ra_image' in kwargs_ps[0]:
            return 0
        x_image = kwargs_ps[0]['ra_image']
        y_image = kwargs_ps[0]['dec_image']
        ra_image_list, dec_image_list = self.imSim.image_positions(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        num_image = len(ra_image_list[0])
        if num_image != len(x_image):
            return -10**15
        dist = util.min_square_dist(ra_image_list[0], dec_image_list[0], x_image, y_image)
        logL = -np.sum(dist/sigma**2)/2
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

    def check_positive_flux(self, kwargs_source, kwargs_lens_light, kwargs_ps):
        penalty = 0
        pos_bool = self.imSim.PointSource.check_positive_flux(kwargs_ps)
        if pos_bool is False:
            penalty += 10**15
        pos_bool = self.imSim.SourceModel.check_positive_flux_profile(kwargs_source)
        if pos_bool is False:
            penalty += 10**15
        pos_bool = self.imSim.LensLightModel.check_positive_flux_profile(kwargs_lens_light)
        if pos_bool is False:
            penalty += 10 ** 15
        return penalty

    def logL_delay(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """
        routine to compute the log likelihood of the time delay distance
        :param args:
        :return:
        """
        delay_arcsec = self.imSim.fermat_potential(kwargs_lens, kwargs_ps)
        D_dt_model = kwargs_cosmo['D_dt']
        delay_days = const.delay_arcsec2days(delay_arcsec[0], D_dt_model)
        logL = self._logL_delays(delay_days, self._delays_measured, self._delays_errors)
        return logL

    def _logL_delays(self, delays_model, delays_measured, delays_errors):
        """
        log likelihood of modeled delays vs measured time delays under considerations of errors

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
        n = self.imSim.numData_evaluate(compute_bool=self._compute_bool)
        num_param, _ = self.param.num_param()
        num_linear = 0
        for bool in self._compute_bool:
            if bool is True:
                num_linear += self.param.num_param_linear()
        return n - num_param - num_linear

    def __call__(self, a):
        return self.logL(a)

    def likelihood(self, a):
        return self.logL(a)

    def computeLikelihood(self, ctx):
        logL, _ = self.logL(ctx.getParams())
        return logL

    def setup(self):
        pass
