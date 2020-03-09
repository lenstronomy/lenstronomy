__author__ = 'sibirrer'


from lenstronomy.Sampling.Likelihoods.time_delay_likelihood import TimeDelayLikelihood
from lenstronomy.Sampling.Likelihoods.image_likelihood import ImageLikelihood
from lenstronomy.Sampling.Likelihoods.position_likelihood import PositionLikelihood
from lenstronomy.Sampling.Likelihoods.flux_ratio_likelihood import FluxRatioLikelihood
from lenstronomy.Sampling.Likelihoods.prior_likelihood import PriorLikelihood
import lenstronomy.Util.class_creator as class_creator
import numpy as np


class LikelihoodModule(object):
    """
    this class contains the routines to run a MCMC process
    the key components are:
    - imSim_class: an instance of a class that simulates one (or more) images and returns the likelihood, such as
    ImageModel(), Multiband(), MultiExposure()
    - param_class: instance of a Param() class that can cast the sorted list of parameters that are sampled into the conventions of the imSim_class

    Additional arguments are supported for adding a time-delay likelihood etc (see __init__ definition)
    """
    def __init__(self, kwargs_data_joint, kwargs_model, param_class, image_likelihood=True, check_bounds=True,
                 check_matched_source_position=False, astrometric_likelihood=False, image_position_likelihood=False,
                 source_position_likelihood=False, image_position_uncertainty=0.004, check_positive_flux=False,
                 source_position_tolerance=0.001, source_position_sigma=0.001, force_no_add_image=False,
                 source_marg=False, linear_prior=None, restrict_image_number=False,
                 max_num_images=None, bands_compute=None, time_delay_likelihood=False,
                 force_minimum_source_surface_brightness=False, flux_min=0, image_likelihood_mask_list=None,
                 flux_ratio_likelihood=False, kwargs_flux_compute={}, prior_lens=[], prior_source=[],
                 prior_extinction=[], prior_lens_light=[], prior_ps=[], prior_special=[], prior_lens_kde=[],
                 prior_source_kde=[], prior_lens_light_kde=[], prior_ps_kde=[], prior_special_kde=[],
                 prior_extinction_kde=[], prior_lens_lognormal=[], prior_source_lognormal=[],
                 prior_extinction_lognormal=[], prior_lens_light_lognormal=[], prior_ps_lognormal=[],
                 prior_special_lognormal=[], custom_logL_addition=None):
        """
        initializing class


        :param param_class: instance of a Param() class that can cast the sorted list of parameters that are sampled into the
        conventions of the imSim_class
        :param image_likelihood: bool, option to compute the imaging likelihood
        :param source_position_likelihood: bool, if True, ray-traces image positions back to source plane and evaluates
        relative errors in respect ot the position_uncertainties in the image plane
        :param check_bounds:  bool, option to punish the hard bounds in parameter space
        :param check_matched_source_position: bool, option to check whether point source position solver finds a solution to match all
         the image positions in the same source plane coordinate
        :param astrometric_likelihood: bool, additional likelihood term of the predicted vs modelled point source position
        :param flaot, image_position_uncertainty: 1-sigma Gaussian uncertainty on the point source position
        (only used if point_source_likelihood=True)
        :param check_positive_flux: bool, option to punish models that do not have all positive linear amplitude parameters
        :param source_position_tolerance: float, punishment of check_solver occurs when image positions are predicted further
        away than this number
        :param image_likelihood_mask_list: list of boolean 2d arrays of size of images marking the pixels to be evaluated in the likelihood
        :param force_no_add_image: bool, if True: computes ALL image positions of the point source. If there are more
        images predicted than modelled, a punishment occures
        :param source_marg: marginalization addition on the imaging likelihood based on the covariance of the infered
        linear coefficients
        :param linear_prior: float or list of floats (when multi-linear setting is chosen) indicating the range of
        linear amplitude priors when computing the marginalization term.
        :param restrict_image_number: bool, if True: computes ALL image positions of the point source. If there are more
        images predicted than indicated in max_num_images, a punishment occurs
        :param max_num_images: int, see restrict_image_number
        :param bands_compute: list of bools with same length as data objects, indicates which "band" to include in the fitting
        :param time_delay_likelihood: bool, if True computes the time-delay likelihood of the FIRST point source
        :param force_minimum_source_surface_brightness: bool, if True, evaluates the source surface brightness on a grid
        and evaluates if all positions have positive flux
        :param kwargs_flux_compute: keyword arguments of how to compute the image position fluxes (see FluxRatioLikeliood)
        :param custom_logL_addition: a definition taking as arguments (kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special, kwargs_extinction)
        and returns a logL (punishing) value.
        """
        multi_band_list, image_type, time_delays_measured, time_delays_uncertainties, flux_ratios, flux_ratio_errors, ra_image_list, dec_image_list = self._unpack_data(**kwargs_data_joint)
        if len(multi_band_list) == 0:
            image_likelihood = False

        self.param = param_class
        self._lower_limit, self._upper_limit = self.param.param_limits()
        lens_model_class, source_model_class, lens_light_model_class, point_source_class, extinction_class = class_creator.create_class_instances(**kwargs_model)
        self.PointSource = point_source_class

        self._prior_likelihood = PriorLikelihood(prior_lens, prior_source, prior_lens_light, prior_ps, prior_special, prior_extinction,
                                                 prior_lens_kde, prior_source_kde, prior_lens_light_kde, prior_ps_kde,
                                                 prior_special_kde, prior_extinction_kde,
                                                 prior_lens_lognormal, prior_source_lognormal,
                                                 prior_lens_light_lognormal, prior_ps_lognormal,
                                                 prior_special_lognormal, prior_extinction_lognormal,
                                                 )
        self._time_delay_likelihood = time_delay_likelihood
        if self._time_delay_likelihood is True:
            self.time_delay_likelihood = TimeDelayLikelihood(time_delays_measured, time_delays_uncertainties,
                                                             lens_model_class, point_source_class)

        self._image_likelihood = image_likelihood
        if self._image_likelihood is True:
            self.image_likelihood = ImageLikelihood(multi_band_list, image_type, kwargs_model, bands_compute=bands_compute,
                                                    likelihood_mask_list=image_likelihood_mask_list,
                                                    source_marg=source_marg, linear_prior=linear_prior,
                                                    force_minimum_source_surface_brightness=force_minimum_source_surface_brightness,
                                                    flux_min=flux_min)
        self._position_likelihood = PositionLikelihood(point_source_class, astrometric_likelihood=astrometric_likelihood,
                                                       image_position_likelihood=image_position_likelihood,
                                                       source_position_likelihood=source_position_likelihood,
                                                       ra_image_list=ra_image_list, dec_image_list=dec_image_list,
                                                       image_position_uncertainty=image_position_uncertainty,
                                                       check_matched_source_position=check_matched_source_position,
                                                       source_position_tolerance=source_position_tolerance,
                                                       source_position_sigma=source_position_sigma,
                                                       force_no_add_image=force_no_add_image,
                                                       restrict_image_number=restrict_image_number,
                                                       max_num_images=max_num_images)
        self._flux_ratio_likelihood = flux_ratio_likelihood
        self._kwargs_flux_compute = kwargs_flux_compute
        if self._flux_ratio_likelihood is True:
            self.flux_ratio_likelihood = FluxRatioLikelihood(lens_model_class, flux_ratios, flux_ratio_errors,
                                                             **self._kwargs_flux_compute)
        self._check_positive_flux = check_positive_flux
        self._check_bounds = check_bounds
        self._custom_logL_addition = custom_logL_addition

    def _unpack_data(self, multi_band_list=[], multi_band_type='multi-linear', time_delays_measured=None,
                     time_delays_uncertainties=None, flux_ratios=None, flux_ratio_errors=None, ra_image_list=[], dec_image_list=[]):
        """

        :param multi_band_list: list of [[kwargs_data, kwargs_psf, kwargs_numerics], [], ...]
        :param multi_band_type: string, type of multi-plane settings (multi-linear or joint-linear)
        :param time_delays_measured: measured time delays (units of days)
        :param time_delays_uncertainties: uncertainties in time-delay measurement
        :param flux_ratios: flux ratios of point sources
        :param flux_ratio_errors: error in flux ratio measurement
        :return:
        """
        return multi_band_list, multi_band_type, time_delays_measured, time_delays_uncertainties, flux_ratios, flux_ratio_errors, ra_image_list, dec_image_list

    def _reset_point_source_cache(self, bool=True):
        self.PointSource.delete_lens_model_cache()
        self.PointSource.set_save_cache(bool)
        if self._image_likelihood is True:
            self.image_likelihood.reset_point_source_cache(bool)

    def logL(self, args, verbose=False):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chain
        """
        #extract parameters
        kwargs_return = self.param.args2kwargs(args)
        if self._check_bounds is True:
            penalty, bound_hit = self.check_bounds(args, self._lower_limit, self._upper_limit, verbose=verbose)
            if bound_hit is True:
                #print(-penalty, 'test penalty')
                return -np.inf
        return self.log_likelihood(kwargs_return, verbose=verbose)

    def log_likelihood(self, kwargs_return, verbose=False):
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special = kwargs_return['kwargs_lens'], \
                                                                                   kwargs_return['kwargs_source'], \
                                                                                   kwargs_return['kwargs_lens_light'], \
                                                                                   kwargs_return['kwargs_ps'], \
                                                                                   kwargs_return['kwargs_special']
        #generate image and computes likelihood
        self._reset_point_source_cache(bool=True)
        logL = 0

        if self._image_likelihood is True:
            logL_image = self.image_likelihood.logL(**kwargs_return)
            logL += logL_image
            if verbose is True:
                print('image logL = %s' % logL_image)
        if self._time_delay_likelihood is True:
            logL_time_delay = self.time_delay_likelihood.logL(kwargs_lens, kwargs_ps, kwargs_special)
            logL += logL_time_delay
            if verbose is True:
                print('time-delay logL = %s' % logL_time_delay)
        if self._check_positive_flux is True:
            bool = self.param.check_positive_flux(kwargs_source, kwargs_lens_light, kwargs_ps)
            if bool is False:
                logL -= 10**5
                if verbose is True:
                    print('non-positive surface brightness parameters detected!')
        if self._flux_ratio_likelihood is True:
            ra_image_list, dec_image_list = self.PointSource.image_position(kwargs_ps=kwargs_ps,
                                                                            kwargs_lens=kwargs_lens)
            #x_pos, y_pos = self.param.real_image_positions(ra_image_list[0], dec_image_list[0], kwargs_special)
            x_pos, y_pos = ra_image_list[0], dec_image_list[0]
            logL_flux_ratios = self.flux_ratio_likelihood.logL(x_pos, y_pos, kwargs_lens, kwargs_special)
            logL += logL_flux_ratios
            if verbose is True:
                print('flux ratio logL = %s' % logL_flux_ratios)
        logL += self._position_likelihood.logL(kwargs_lens, kwargs_ps, kwargs_special, verbose=verbose)
        logL_prior = self._prior_likelihood.logL(**kwargs_return)
        logL += logL_prior
        if verbose is True:
            print('Prior likelihood = %s' % logL_prior)
        if self._custom_logL_addition is not None:
            logL_cond = self._custom_logL_addition(**kwargs_return)
            logL += logL_cond
            if verbose is True:
                print('custom added logL = %s' % logL_cond)
        self._reset_point_source_cache(bool=False)
        return logL#, None

    @staticmethod
    def check_bounds(args, lowerLimit, upperLimit, verbose=False):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        penalty = 0.
        bound_hit = False
        for i in range(0, len(args)):
            if args[i] < lowerLimit[i] or args[i] > upperLimit[i]:
                penalty = 10.**5
                bound_hit = True
                if verbose is True:
                    print('parameter %s with value %s hit the bounds [%s, %s] ' % (i, args[i], lowerLimit[i], upperLimit[i]))
                return penalty, bound_hit
        return penalty, bound_hit

    @property
    def num_data(self):
        """

        :return: number of independent data points in the combined fitting
        """
        num_data = 0
        if self._image_likelihood is True:
            num_data += self.image_likelihood.num_data
        if self._time_delay_likelihood is True:
            num_data += self.time_delay_likelihood.num_data
        if self._flux_ratio_likelihood is True:
            num_data += self.flux_ratio_likelihood.num_data
        num_data += self._position_likelihood.num_data
        return num_data

    @property
    def param_limits(self):
        return self._lower_limit, self._upper_limit

    def effectiv_num_data_points(self, **kwargs):
        """
        returns the effective number of data points considered in the X2 estimation to compute the reduced X2 value
        """
        num_linear = 0
        if self._image_likelihood is True:
            num_linear = self.image_likelihood.num_param_linear(**kwargs)
        num_param, param_names = self.param.num_param()
        return self.num_data - num_param - num_linear

    def __call__(self, a):
        return self.logL(a)

    def negativelogL(self, a):
        logL = self.logL(a)
        return -logL

    def likelihood(self, a):
        return self.logL(a)

    def likelihood_derivative(self, a):
        """

        :param a: array
        :return: logL, derivative estimatoe (None)
        """
        return self.logL(a), None
