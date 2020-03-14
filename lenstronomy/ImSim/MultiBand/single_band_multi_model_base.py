from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.ImSim.image_sparse_solve import ImageSparseFit
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import class_creator


class SingleBandMultiModelBase(object):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    decoupled linear parameters and the option to pass/select different light models for the different bands

    the class instance needs to have a forth row in the multi_band_list with keyword arguments 'source_light_model_index' and
    'lens_light_model_index' as bool arrays of the size of the total source model types and lens light model types,
    specifying which model is evaluated for which band.

    """

    def __init__(self, multi_band_list, kwargs_model, likelihood_mask_list=None, band_index=0):
        index_lens_model_list = kwargs_model.get('index_lens_model_list', [None for i in range(len(multi_band_list))])
        self._index_lens_model = index_lens_model_list[band_index]
        index_source_list = kwargs_model.get('index_source_light_model_list', [None for i in range(len(multi_band_list))])
        self._index_source = index_source_list[band_index]
        index_lens_light_list = kwargs_model.get('index_lens_light_model_list', [None for i in range(len(multi_band_list))])
        self._index_lens_light = index_lens_light_list[band_index]
        index_point_source_list = kwargs_model.get('index_point_source_model_list', [None for i in range(len(multi_band_list))])
        self._index_point_source = index_point_source_list[band_index]
        index_optical_depth = kwargs_model.get('index_optical_depth_model_list',
                                                   [None for i in range(len(multi_band_list))])
        self._index_optical_depth = index_optical_depth[band_index]

    def num_param_linear(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None):
        """

        :param compute_bool:
        :return: number of linear coefficients to be solved for in the linear inversion
        """
        kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i = self.select_kwargs(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
        num = self._num_param_linear(kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i)
        return num

    def error_map_source(self, kwargs_source, x_grid, y_grid, cov_param):
        """
        variance of the linear source reconstruction in the source plane coordinates,
        computed by the diagonal elements of the covariance matrix of the source reconstruction as a sum of the errors
        of the basis set.

        :param kwargs_source: keyword arguments of source model
        :param x_grid: x-axis of positions to compute error map
        :param y_grid: y-axis of positions to compute error map
        :param cov_param: covariance matrix of liner inversion parameters
        :return: diagonal covariance errors at the positions (x_grid, y_grid)
        """
        if self._index_source is None:
            kwargs_source_i = kwargs_source
        else:
            kwargs_source_i = [kwargs_source[k] for k in self._index_source]
        return self._error_map_source(kwargs_source_i, x_grid, y_grid, cov_param)

    def select_kwargs(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                      kwargs_extinction=None, kwargs_special=None):
        """
        select subset of kwargs lists referenced to this imaging band

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        if self._index_lens_model is None:
            kwargs_lens_i = kwargs_lens
        else:
            kwargs_lens_i = [kwargs_lens[k] for k in self._index_lens_model]
        if self._index_source is None:
            kwargs_source_i = kwargs_source
        else:
            kwargs_source_i = [kwargs_source[k] for k in self._index_source]
        if self._index_lens_light is None:
            kwargs_lens_light_i = kwargs_lens_light
        else:
            kwargs_lens_light_i = [kwargs_lens_light[k] for k in self._index_lens_light]
        if self._index_point_source is None:
            kwargs_ps_i = kwargs_ps
        else:
            kwargs_ps_i = [kwargs_ps[k] for k in self._index_point_source]
        if self._index_optical_depth is None or kwargs_extinction is None:
            kwargs_extinction_i = kwargs_extinction
        else:
            kwargs_extinction_i = [kwargs_extinction[k] for k in self._index_optical_depth]
        return kwargs_lens_i, kwargs_source_i, kwargs_lens_light_i, kwargs_ps_i, kwargs_extinction_i

    def select_image_fit(self, multi_band_list, kwargs_model, likelihood_mask_list=None, band_index=0):
        if likelihood_mask_list is None:
            likelihood_mask = None
        else:
            likelihood_mask = likelihood_mask_list[band_index]
        lens_model_class, source_model_class, lens_light_model_class, point_source_class, extinction_class = class_creator.create_class_instances(band_index=band_index, **kwargs_model)
        kwargs_data = multi_band_list[band_index][0]
        kwargs_psf = multi_band_list[band_index][1]
        kwargs_numerics = multi_band_list[band_index][2]
        data_i = ImageData(**kwargs_data)
        psf_i = PSF(**kwargs_psf)
        return data_i, psf_i, lens_model_class, source_model_class, lens_light_model_class, point_source_class, extinction_class, kwargs_numerics, likelihood_mask
