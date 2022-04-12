import copy
import numpy as np

import lenstronomy.Util.class_creator as class_creator
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
class MultiBandImageReconstruction(object):
    """
    this class manages the output/results of a fitting process and can conveniently access image reconstruction
    properties in multi-band fitting.
    In particular, the fitting result does not come with linear inversion parameters (which may or may not be joint
    or different for multiple bands) and this class performs the linear inversion for the surface brightness amplitudes
    and stores them for each individual band to be accessible by the user.

    This class is is the backbone of the ModelPlot routine that provides the interface of this class with plotting and
    illustration routines.
    """

    def __init__(self, multi_band_list, kwargs_model, kwargs_params, multi_band_type='multi-linear',
                 kwargs_likelihood=None, verbose=True):
        """

        :param multi_band_list: list of imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :param kwargs_model: model keyword argument list
        :param kwargs_params: keyword arguments of the model parameters, same as output of FittingSequence() 'kwargs_result'
        :param multi_band_type: string, option when having multiple imaging data sets modelled simultaneously. Options are:
            - 'multi-linear': linear amplitudes are inferred on single data set
            - 'linear-joint': linear amplitudes ae jointly inferred
            - 'single-band': single band
        :param kwargs_likelihood: likelihood keyword arguments as supported by the Likelihood() class
        :param verbose: if True (default), computes and prints the total log-likelihood.
        This can deactivated for speedup purposes (does not run linear inversion again), and reduces the number of prints.
        """
        # here we retrieve those settings in the likelihood keyword arguments that are relevant for the image reconstruction
        if kwargs_likelihood is None:
            kwargs_likelihood = {}
        image_likelihood_mask_list = kwargs_likelihood.get('image_likelihood_mask_list', None)
        source_marg = kwargs_likelihood.get('source_marg', False)
        linear_prior = kwargs_likelihood.get('linear_prior', None)
        bands_compute = kwargs_likelihood.get('bands_compute', None)
        if bands_compute is None:
            bands_compute = [True] * len(multi_band_list)
        if multi_band_type == 'single-band':
            multi_band_type = 'multi-linear'  # this makes sure that the linear inversion outputs are coming in a list
        self._imageModel = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model,
                                                       bands_compute=bands_compute,
                                                       likelihood_mask_list=image_likelihood_mask_list)

        # here we perform the (joint) linear inversion with all data
        model, error_map, cov_param, param = self._imageModel.image_linear_solve(inv_bool=True, **kwargs_params)
        check_solver_error(param)

        if verbose:
            logL = self._imageModel.likelihood_data_given_model(source_marg=source_marg, linear_prior=linear_prior, **kwargs_params)
            n_data = self._imageModel.num_data_evaluate
            if n_data > 0:
                print(logL * 2 / n_data, 'reduced X^2 of all evaluated imaging data combined.')

        self.model_band_list = []
        for i in range(len(multi_band_list)):
            if bands_compute[i] is True:
                if multi_band_type == 'joint-linear':
                    param_i = param
                    cov_param_i = cov_param
                else:
                    param_i = param[i]
                    cov_param_i = cov_param[i]

                model_band = ModelBand(multi_band_list, kwargs_model, model[i], error_map[i], cov_param_i,
                                       param_i, copy.deepcopy(kwargs_params),
                                       image_likelihood_mask_list=image_likelihood_mask_list, band_index=i,
                                       verbose=verbose)
                self.model_band_list.append(model_band)
            else:
                self.model_band_list.append(None)

    def band_setup(self, band_index=0):
        """
        ImageModel() instance and keyword arguments of the model components to execute all the options of the ImSim
         core modules.

        :param band_index: integer (>=0) of imaging band in order of multi_band_list input to this class
        :return: ImageModel() instance and keyword arguments of the model
        """
        i = int(band_index)
        if self.model_band_list[i] is None:
            raise ValueError("band %s is not computed or out of range." % i)
        return self.model_band_list[i].image_model_class, self.model_band_list[i].kwargs_model


@export
class ModelBand(object):
    """
    class to plot a single band given the full modeling results
    This class has it's specific role when the linear inference is performed on the joint band level and/or when only
    a subset of model components get used for this specific band in the modeling.

    """
    def __init__(self, multi_band_list, kwargs_model, model, error_map, cov_param, param, kwargs_params,
                 image_likelihood_mask_list=None, band_index=0, verbose=True):
        """

        :param multi_band_list: list of imaging data configuration [[kwargs_data, kwargs_psf, kwargs_numerics], [...]]
        :param kwargs_model: model keyword argument list for the full multi-band modeling
        :param model: 2d numpy array of modeled image for the specified band
        :param error_map: 2d numpy array of size of the image, additional error in the pixels coming from PSF uncertainties
        :param cov_param: covariance matrix of the linear inversion
        :param param: 1d numpy array of the linear coefficients of this imaging band
        :param kwargs_params: keyword argument of keyword argument lists of the different model components selected for
         the imaging band, NOT including linear amplitudes (not required as being overwritten by the param list)
        :param image_likelihood_mask_list: list of 2d numpy arrays of likelihood masks (for all bands)
        :param band_index: integer of the band to be considered in this class
        :param verbose: if True (default), prints the reduced chi2 value for the current band.
        """

        self._bandmodel = SingleBandMultiModel(multi_band_list, kwargs_model, likelihood_mask_list=image_likelihood_mask_list,
                                               band_index=band_index)
        self._kwargs_special_partial = kwargs_params.get('kwargs_special', None)
        kwarks_lens_partial, kwargs_source_partial, kwargs_lens_light_partial, kwargs_ps_partial, self._kwargs_extinction_partial = self._bandmodel.select_kwargs(**kwargs_params)
        self._kwargs_lens_partial, self._kwargs_source_partial, self._kwargs_lens_light_partial, self._kwargs_ps_partial = self._bandmodel.update_linear_kwargs(param, kwarks_lens_partial, kwargs_source_partial, kwargs_lens_light_partial, kwargs_ps_partial)
        # this is an (out-commented) example of how to re-create the model in this band
        #model_new = self.bandmodel.image(self._kwargs_lens_partial, self._kwargs_source_partial, self._kwargs_lens_light_partial, self._kwargs_ps_partial, self._kwargs_special_partial, self._kwargs_extinction_partial)

        self._norm_residuals = self._bandmodel.reduced_residuals(model, error_map=error_map)
        self._reduced_x2 = self._bandmodel.reduced_chi2(model, error_map=error_map)
        if verbose:
            print("reduced chi^2 of data ", band_index, "= ", self._reduced_x2)

        self._model = model
        self._cov_param = cov_param
        self._param = param
        self._error_map = error_map

    @property
    def model(self):
        """

        :return: model, 2d numpy array
        """
        return self._model

    @property
    def norm_residuals(self):
        """

        :return: normalized residuals, 2d numpy array
        """
        return self._norm_residuals

    @property
    def image_model_class(self):
        """
        ImageModel() class instance of the single band with only the model components applied to this band

        :return: SingleBandMultiModel() instance, which inherits the ImageModel instance
        """
        return self._bandmodel

    @property
    def kwargs_model(self):
        """

        :return: keyword argument of keyword argument lists of the different model components selected for the imaging
         band, including linear amplitudes. These format matches the image_model_class() return
        """
        kwargs_return = {'kwargs_lens': self._kwargs_lens_partial, 'kwargs_source': self._kwargs_source_partial,
                         'kwargs_lens_light': self._kwargs_lens_light_partial, 'kwargs_ps': self._kwargs_ps_partial,
                         'kwargs_special': self._kwargs_special_partial,
                         'kwargs_extinction': self._kwargs_extinction_partial}
        return kwargs_return


@export
def check_solver_error(image):
    """

    :param image: numpy array of modelled image from linear inversion
    :return: bool, True if solver could not find a unique solution, False if solver works
    """
    result = np.all(image == 0)
    if result:
        Warning('Linear inversion of surface brightness components did not result in a unique solution.'
                'All linear amplitude parameters are set =0 instead. Please check whether '
                'a) there are too many basis functions in the model, '
                'or b) some linear basis sets are outside of the image/likelihood mask.')
    return result
