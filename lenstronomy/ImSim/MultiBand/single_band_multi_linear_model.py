from lenstronomy.ImSim.MultiBand.single_band_multi_model_base import SingleBandMultiModelBase
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import class_creator


class SingleBandMultiLinearModel(SingleBandMultiModelBase, ImageLinearFit):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    decoupled linear parameters and the option to pass/select different light models for the different bands

    the class instance needs to have a forth row in the multi_band_list with keyword arguments 'source_light_model_index' and
    'lens_light_model_index' as bool arrays of the size of the total source model types and lens light model types,
    specifying which model is evaluated for which band.

    """

    def __init__(self, multi_band_list, kwargs_model, likelihood_mask_list=None, band_index=0):
        self.type = 'single-band-multi-linear-model'
        SingleBandMultiModelBase.__init__(self, multi_band_list, kwargs_model, 
                                          likelihood_mask_list=likelihood_mask_list, 
                                          band_index=band_index)
        data_i, psf_i, lens_model_class, source_model_class, lens_light_model_class, \
            point_source_class, extinction_class, kwargs_numerics, likelihood_mask \
            = self.select_image_fit(multi_band_list, kwargs_model, likelihood_mask_list=likelihood_mask_list, 
                                    band_index=band_index)
        ImageLinearFit.__init__(self, data_i, psf_i, lens_model_class, source_model_class,
                                lens_light_model_class, point_source_class, extinction_class,
                                kwargs_numerics=kwargs_numerics, likelihood_mask=likelihood_mask)

    def image_linear_solve(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                           kwargs_extinction=None, kwargs_special=None, inv_bool=False):
        """
        computes the image (lens and source surface brightness with a given lens model).
        The linear parameters are computed with a weighted linear least square optimization (i.e. flux normalization of the brightness profiles)
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the purpose of the covariance matrix.
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        wls_model, error_map, cov_param, param = self._image_linear_solve(kwargs_lens_i, kwargs_source_i,
                                                                          kwargs_lens_light_i, kwargs_ps_i,
                                                                          kwargs_extinction_i, kwargs_special, inv_bool=inv_bool)
        return wls_model, error_map, cov_param, param

    def likelihood_data_given_model(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                                    kwargs_extinction=None, kwargs_special=None, source_marg=False, linear_prior=None):
        """
        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return: log likelihood (natural logarithm) (sum of the log likelihoods of the individual images)
        """
        # generate image
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        logL = self._likelihood_data_given_model(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i,
                                                 kwargs_extinction_i, kwargs_special, source_marg=source_marg,
                                                 linear_prior=linear_prior)
        return logL

    def linear_response_matrix(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                               kwargs_extinction=None, kwargs_special=None):
        """
        computes the linear response matrix (m x n), with n beeing the data size and m being the coefficients

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens,
                                                                                              kwargs_source,
                                                                                              kwargs_lens_light,
                                                                                              kwargs_ps,
                                                                                              kwargs_extinction)
        A = self._linear_response_matrix(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i,
                                         kwargs_extinction_i, kwargs_special)
        return A
