__author__ = 'sibirrer'

from lenstronomy.Sampling.Likelihoods.time_delay_likelihood import TimeDelayLikelihood
from lenstronomy.Sampling.Likelihoods.image_likelihood import ImageLikelihood
from lenstronomy.Sampling.Likelihoods.position_likelihood import PositionLikelihood
from lenstronomy.Sampling.Likelihoods.flux_ratio_likelihood import FluxRatioLikelihood
from lenstronomy.Sampling.Likelihoods.prior_likelihood import PriorLikelihood
from lenstronomy.Sampling.Likelihoods.kinematic_2D_likelihood import KinLikelihood
import lenstronomy.Util.class_creator as class_creator
import numpy as np

__all__ = ['LikelihoodModule']


class LikelihoodModule(object):
    """
    this class contains the routines to run a MCMC process
    the key components are:
    - imSim_class: an instance of a class that simulates one (or more) images and returns the likelihood, such as
    ImageModel(), Multiband(), MultiExposure()
    - param_class: instance of a Param() class that can cast the sorted list of parameters that are sampled into the
    conventions of the imSim_class

    Additional arguments are supported for adding a time-delay likelihood etc (see __init__ definition)
    """
    def __init__(self, kwargs_data_joint, kwargs_model, param_class, image_likelihood=True, check_bounds=True,
                 check_matched_source_position=False, astrometric_likelihood=False, image_position_likelihood=False,
                 source_position_likelihood=False, image_position_uncertainty=0.004, check_positive_flux=False,
                 source_position_tolerance=0.001, source_position_sigma=0.001, force_no_add_image=False,
                 source_marg=False, linear_prior=None, restrict_image_number=False,
                 max_num_images=None, bands_compute=None, time_delay_likelihood=False,
                 image_likelihood_mask_list=None,
                 flux_ratio_likelihood=False, kwargs_flux_compute=None, prior_lens=None, prior_source=None,
                 prior_extinction=None, prior_lens_light=None, prior_ps=None, prior_special=None, prior_lens_kde=None,
                 prior_source_kde=None, prior_lens_light_kde=None, prior_ps_kde=None, prior_special_kde=None,
                 prior_extinction_kde=None, prior_lens_lognormal=None, prior_source_lognormal=None,
                 prior_extinction_lognormal=None, prior_lens_light_lognormal=None, prior_ps_lognormal=None,
                 prior_special_lognormal=None, custom_logL_addition=None, kwargs_pixelbased=None,
                 kinematic_2D_likelihood=False, kin_lens_idx=0, kin_lens_light_idx=0):
        """
        initializing class


        :param param_class: instance of a Param() class that can cast the sorted list of parameters that are sampled
         into the conventions of the imSim_class
        :param image_likelihood: bool, option to compute the imaging likelihood
        :param source_position_likelihood: bool, if True, ray-traces image positions back to source plane and evaluates
         relative errors in respect ot the position_uncertainties in the image plane
        :param check_bounds:  bool, option to punish the hard bounds in parameter space
        :param check_matched_source_position: bool, option to check whether point source position of solver finds a
         solution to match all the image positions in the same source plane coordinate
        :param astrometric_likelihood: bool, additional likelihood term of the predicted vs modelled point source
         position
        :param image_position_uncertainty: float, 1-sigma Gaussian uncertainty on the point source position
         (only used if point_source_likelihood=True)
        :param check_positive_flux: bool, option to punish models that do not have all positive linear amplitude
         parameters
        :param source_position_tolerance: float, punishment of check_solver occurs when image positions are predicted
         further away than this number
        :param image_likelihood_mask_list: list of boolean 2d arrays of size of images marking the pixels to be
         evaluated in the likelihood
        :param force_no_add_image: bool, if True: computes ALL image positions of the point source. If there are more
         images predicted than modelled, a punishment occurs
        :param source_marg: marginalization addition on the imaging likelihood based on the covariance of the inferred
         linear coefficients
        :param linear_prior: float or list of floats (when multi-linear setting is chosen) indicating the range of
         linear amplitude priors when computing the marginalization term.
        :param restrict_image_number: bool, if True: computes ALL image positions of the point source. If there are more
         images predicted than indicated in max_num_images, a punishment occurs
        :param max_num_images: int, see restrict_image_number
        :param bands_compute: list of bools with same length as data objects, indicates which "band" to include in the
         fitting
        :param time_delay_likelihood: bool, if True computes the time-delay likelihood of the FIRST point source
        :param kwargs_flux_compute: keyword arguments of how to compute the image position fluxes
         (see FluxRatioLikeliood)
        :param custom_logL_addition: a definition taking as arguments (kwargs_lens, kwargs_source, kwargs_lens_light,
         kwargs_ps, kwargs_special, kwargs_extinction) and returns a logL (punishing) value.
        :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver
         (see SLITronomy documentation)
        :param kinematic_2D_likelihood: bool, option to compute the kinematic likelihood
        """
        multi_band_list, multi_band_type, time_delays_measured, time_delays_uncertainties, flux_ratios, flux_ratio_errors, ra_image_list, dec_image_list, kinematic_data = self._unpack_data(**kwargs_data_joint)
        if len(multi_band_list) == 0:
            image_likelihood = False
        self.kinematic_class = kinematic_data
        self.param = param_class
        self._lower_limit, self._upper_limit = self.param.param_limits()
        self._prior_likelihood = PriorLikelihood(prior_lens, prior_source, prior_lens_light, prior_ps, prior_special,
                                                 prior_extinction,
                                                 prior_lens_kde, prior_source_kde, prior_lens_light_kde, prior_ps_kde,
                                                 prior_special_kde, prior_extinction_kde,
                                                 prior_lens_lognormal, prior_source_lognormal,
                                                 prior_lens_light_lognormal, prior_ps_lognormal,
                                                 prior_special_lognormal, prior_extinction_lognormal,
                                                 )
        self._time_delay_likelihood = time_delay_likelihood
        self._image_likelihood = image_likelihood
        self._flux_ratio_likelihood = flux_ratio_likelihood
        self._kinematic_2D_likelihood = kinematic_2D_likelihood
        if kwargs_flux_compute is None:
            kwargs_flux_compute = {}
        linear_solver = self.param.linear_solver
        self._kwargs_flux_compute = kwargs_flux_compute
        self._check_bounds = check_bounds
        self._custom_logL_addition = custom_logL_addition
        self._kwargs_time_delay = {'time_delays_measured': time_delays_measured,
                                   'time_delays_uncertainties': time_delays_uncertainties}
        self._kwargs_imaging = {'multi_band_list': multi_band_list, 'multi_band_type': multi_band_type,
                               'bands_compute': bands_compute,
                               'image_likelihood_mask_list': image_likelihood_mask_list, 'source_marg': source_marg,
                               'linear_prior': linear_prior, 'check_positive_flux': check_positive_flux,
                               'kwargs_pixelbased': kwargs_pixelbased, 'linear_solver': linear_solver}
        self._kwargs_position = {'astrometric_likelihood': astrometric_likelihood,
                                 'image_position_likelihood': image_position_likelihood,
                                 'source_position_likelihood': source_position_likelihood,
                                 'ra_image_list': ra_image_list, 'dec_image_list': dec_image_list,
                                 'image_position_uncertainty': image_position_uncertainty,
                                 'check_matched_source_position': check_matched_source_position,
                                 'source_position_tolerance': source_position_tolerance,
                                 'source_position_sigma': source_position_sigma,
                                 'force_no_add_image': force_no_add_image,
                                 'restrict_image_number': restrict_image_number, 'max_num_images': max_num_images}
        self._kwargs_flux = {'flux_ratios': flux_ratios, 'flux_ratio_errors': flux_ratio_errors}
        self._kwargs_flux.update(self._kwargs_flux_compute)
        
        if self._kinematic_2D_likelihood is True :
            print("Note that the 2D kinematic likelihood assumes that the lens and lens light have the same center and orientation")
            if len(multi_band_list) > 1 :
                print('Kinematic Likelihood not meant for multiband, using first band by default')
            if kwargs_model['lens_model_list'][kin_lens_idx] not in ['PEMD_Q_PHI']:
                print('Lens for kinematic is not PEMD_Q_PHI, the 2D kinematic likelihood will break.')
            if kwargs_model['lens_light_model_list'][kin_lens_light_idx] not in ['SERSIC_ELLIPSE_Q_PHI']:
                print('Lens light for kinematic is not SERSIC_ELLIPSE_Q_PHI , the 2D kinematic likelihood will break.')
            self._kin_lens_idx = kin_lens_idx
            self._kin_lens_light_idx = kin_lens_light_idx

        self._class_instances(kwargs_model=kwargs_model, kwargs_imaging=self._kwargs_imaging,
                              kwargs_position=self._kwargs_position, kwargs_flux=self._kwargs_flux,
                              kwargs_time_delay=self._kwargs_time_delay, kinematic_data = self.kinematic_class)


    def _class_instances(self, kwargs_model, kwargs_imaging, kwargs_position, kwargs_flux, kwargs_time_delay, kinematic_data):
        """

        :param kwargs_model: lenstronomy model keyword arguments
        :param kwargs_imaging: keyword arguments for imaging likelihood
        :param kwargs_position: keyword arguments for positional likelihood
        :param kwargs_flux: keyword arguments for flux ratio likelihood
        :param kwargs_time_delay: keyword arguments for time delay likelihood
        :return: updated model instances of this class
        """

        # TODO: in case lens model or point source models are only applied on partial images, then this current class
        # has ambiguities when it comes to position likelihood, time-delay likelihood and flux ratio likelihood
        lens_model_class, _, lens_light_model_class, point_source_class, _ = class_creator.create_class_instances(all_models=True,
                                                                                             **kwargs_model)
        self.PointSource = point_source_class

        if self._time_delay_likelihood is True:
            self.time_delay_likelihood = TimeDelayLikelihood(lens_model_class=lens_model_class,
                                                             point_source_class=point_source_class,
                                                             **kwargs_time_delay)

        if self._image_likelihood is True:
            self.image_likelihood = ImageLikelihood(kwargs_model=kwargs_model, **kwargs_imaging)
        self._position_likelihood = PositionLikelihood(point_source_class, **kwargs_position)
        if self._flux_ratio_likelihood is True:
            self.flux_ratio_likelihood = FluxRatioLikelihood(lens_model_class, **kwargs_flux)
        if self._kinematic_2D_likelihood is True:
            self.kinematic_2D_likelihood = KinLikelihood(kinematic_data, lens_model_class, lens_light_model_class,
                                                         kwargs_imaging['multi_band_list'][0][0],self._kin_lens_idx,
                                                         self._kin_lens_light_idx)

    def __call__(self, a):
        return self.logL(a)

    def logL(self, args, verbose=False):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chain


        :param args: ordered parameter values that are being sampled
        :type args: tuple or list of floats
        :param verbose: if True, makes print statements about individual likelihood components
        :type verbose: boolean
        :returns: log likelihood of the data given the model (natural logarithm)
        """
        # extract parameters
        kwargs_return = self.param.args2kwargs(args)
        if self._check_bounds is True:
            penalty, bound_hit = self.check_bounds(args, self._lower_limit, self._upper_limit, verbose=verbose)
            if bound_hit is True:
                return -10**15
        return self.log_likelihood(kwargs_return, verbose=verbose)

    def log_likelihood(self, kwargs_return, verbose=False):
        """


        :param kwargs_return: need to contain 'kwargs_lens', 'kwargs_source', 'kwargs_lens_light', 'kwargs_ps',
         'kwargs_special'. These entries themselves are lists of keyword argument of the parameters entering the model
         to be evaluated
        :type kwargs_return: keyword arguments
        :param verbose: if True, makes print statements about individual likelihood components
        :type verbose: boolean

        :returns:
         - logL (float) log likelihood of the data given the model (natural logarithm)
        """
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special = kwargs_return['kwargs_lens'], \
                                                                                   kwargs_return['kwargs_source'], \
                                                                                   kwargs_return['kwargs_lens_light'], \
                                                                                   kwargs_return['kwargs_ps'], \
                                                                                   kwargs_return['kwargs_special']
        # update model instance in case of changes affecting it (i.e. redshift sampling in multi-plane)
        self._update_model(kwargs_special)
        # generate image and computes likelihood
        self._reset_point_source_cache(bool_input=True)
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
        if self._flux_ratio_likelihood is True:
            ra_image_list, dec_image_list = self.PointSource.image_position(kwargs_ps=kwargs_ps,
                                                                            kwargs_lens=kwargs_lens)
            x_pos, y_pos = ra_image_list[0], dec_image_list[0]
            logL_flux_ratios = self.flux_ratio_likelihood.logL(x_pos, y_pos, kwargs_lens, kwargs_special)
            logL += logL_flux_ratios
            if verbose is True:
                print('flux ratio logL = %s' % logL_flux_ratios)
        if self._kinematic_2D_likelihood is True:
            logL_kinematic_2D = self.kinematic_2D_likelihood.logL(kwargs_lens,kwargs_lens_light,kwargs_special)
            logL += logL_kinematic_2D
            if verbose is True:
                print('kinematic logL = %s' %logL_kinematic_2D)
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
        self._reset_point_source_cache(bool_input=False)
        return logL  # , None

    @staticmethod
    def check_bounds(args, lowerLimit, upperLimit, verbose=False):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        penalty = 0.
        bound_hit = False
        args = np.atleast_1d(args)
        for i in range(0, len(args)):
            if args[i] < lowerLimit[i] or args[i] > upperLimit[i]:
                penalty = 10.**5
                bound_hit = True
                if verbose is True:
                    print('parameter %s with value %s hit the bounds [%s, %s] ' % (i, args[i], lowerLimit[i],
                                                                                   upperLimit[i]))
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

    def effective_num_data_points(self, **kwargs):
        """
        returns the effective number of data points considered in the X2 estimation to compute the reduced X2 value
        """
        num_linear = 0
        if self._image_likelihood is True:
            num_linear = self.image_likelihood.num_param_linear(**kwargs)
        num_param, param_names = self.param.num_param()
        return self.num_data - num_param - num_linear

    def likelihood(self, a):
        return self.logL(a)

    def negativelogL(self, a):
        """
        for minimizer function, the negative value of the logl value is requested

        :param a: array of parameters
        :return: -logL
        """
        return -self.logL(a)

    @staticmethod
    def _unpack_data(multi_band_list=None, multi_band_type='multi-linear', time_delays_measured=None,
                     time_delays_uncertainties=None, flux_ratios=None, flux_ratio_errors=None, ra_image_list=None,
                     dec_image_list=None, kinematic_data=None):
        """

        :param multi_band_list: list of [[kwargs_data, kwargs_psf, kwargs_numerics], [], ...]
        :param multi_band_type: string, type of multi-plane settings (multi-linear or joint-linear)
        :param time_delays_measured: measured time delays (units of days)
        :param time_delays_uncertainties: uncertainties in time-delay measurement
        :param flux_ratios: flux ratios of point sources
        :param flux_ratio_errors: error in flux ratio measurement
        :return:
        """
        if multi_band_list is None:
            multi_band_list = []
        if ra_image_list is None:
            ra_image_list = []
        if dec_image_list is None:
            dec_image_list = []
        return multi_band_list, multi_band_type, time_delays_measured, time_delays_uncertainties, flux_ratios, \
               flux_ratio_errors, ra_image_list, dec_image_list, kinematic_data

    def _reset_point_source_cache(self, bool_input=True):
        self.PointSource.delete_lens_model_cache()
        self.PointSource.set_save_cache(bool_input)
        if self._image_likelihood is True:
            self.image_likelihood.reset_point_source_cache(bool_input)

    def _update_model(self, kwargs_special):
        """
        updates lens model instance of this class (and all class instances related to it) when an update to the
        modeled redshifts of the deflector and/or source planes are made

        :param kwargs_special: keyword arguments from SpecialParam() class return of sampling arguments
        :return: None, all class instances updated to recent modek
        """
        kwargs_model, update_bool = self.param.update_kwargs_model(kwargs_special)
        if update_bool is True:
            self._class_instances(kwargs_model=kwargs_model, kwargs_imaging=self.kwargs_imaging,
                                  kwargs_position=self._kwargs_position, kwargs_flux=self._kwargs_flux,
                                  kwargs_time_delay=self._kwargs_time_delay)
        # TODO remove redundancies with Param() calls updates
