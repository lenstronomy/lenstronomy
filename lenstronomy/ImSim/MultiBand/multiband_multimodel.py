from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.MultiBand.multi_data_base import MultiDataBase


class MultiBandMultiModel(MultiDataBase):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    decoupled linear parameters and the option to pass/select different light models for the different bands

    the class instance needs to have a forth row in the multi_band_list with keyword arguments 'source_light_model_index' and
    'lens_light_model_index' as bool arrays of the size of the total source model types and lens light model types,
    specifying which model is evaluated for which band.

    """

    def __init__(self, multi_band_list, lens_model_class=None, source_model_list=None, lens_light_model_list=None,
                 point_source_class=None):
        self.type = 'multi-band-multi-model'
        imageModel_list = []
        self._index_source_list = []
        self._index_lens_light_list = []
        for i in range(len(multi_band_list)):
            kwargs_data = multi_band_list[i][0]
            kwargs_psf = multi_band_list[i][1]
            kwargs_numerics = multi_band_list[i][2]
            data_i = Data(kwargs_data=kwargs_data)
            psf_i = PSF(kwargs_psf=kwargs_psf)

            index_source_list = multi_band_list[i][3].get('index_source_light_model', [k for k in range(len(source_model_list))])
            self._index_source_list.append(index_source_list)
            source_model_list_sub = [source_model_list[k] for k in index_source_list]
            source_model_class = LightModel(light_model_list=source_model_list_sub)

            index_lens_light_list = multi_band_list[i][3].get('index_lens_light_model', [k for k in range(len(source_model_list))])
            self._index_lens_light_list.append(index_lens_light_list)
            lens_light_model_list_sub = [lens_light_model_list[k] for k in index_lens_light_list]
            lens_light_model_class = LightModel(light_model_list=lens_light_model_list_sub)

            imageModel = ImageModel(data_i, psf_i, lens_model_class, source_model_class,
                                                    lens_light_model_class, point_source_class,
                                                    kwargs_numerics=kwargs_numerics)
            imageModel_list.append(imageModel)
        super(MultiBandMultiModel, self).__init__(imageModel_list)

    def image_linear_solve(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, inv_bool=False):
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
        wls_list, error_map_list, cov_param_list, param_list = [], [], [], []
        for i in range(self._num_bands):
            kwargs_source_i = [kwargs_source[k] for k in self._index_source_list[i]]
            kwargs_lens_light_i = [kwargs_lens_light[k] for k in self._index_lens_light_list[i]]
            wls_model, error_map, cov_param, param = self._imageModel_list[i].image_linear_solve(kwargs_lens,
                                                                                                 kwargs_source_i,
                                                                                                 kwargs_lens_light_i,
                                                                                                 kwargs_ps,
                                                                                                 inv_bool=inv_bool)
            wls_list.append(wls_model)
            error_map_list.append(error_map)
            cov_param_list.append(cov_param)
            param_list.append(param)
        return wls_list, error_map_list, cov_param_list, param_list

    def likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, source_marg=False,
                                    compute_bool=None):
        """
        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return: log likelihood (natural logarithm) (sum of the log likelihoods of the individual images)
        """
        if compute_bool is None:
            compute_bool = [True] * self._num_bands
        else:
            if not len(compute_bool) == self._num_bands:
                raise ValueError('compute_bool statement has not the same range as number of bands available!')
        # generate image
        logL = 0
        for i in range(self._num_bands):
            if compute_bool[i] is True:
                kwargs_source_i = [kwargs_source[k] for k in self._index_source_list[i]]
                kwargs_lens_light_i = [kwargs_lens_light[k] for k in self._index_lens_light_list[i]]
                logL += self._imageModel_list[i].likelihood_data_given_model(kwargs_lens, kwargs_source_i,
                                                                             kwargs_lens_light_i, kwargs_ps,
                                                                             source_marg=source_marg)
        return logL

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, compute_bool=None):
        """

        :param compute_bool:
        :return: number of linear coefficients to be solved for in the linear inversion
        """
        num = 0
        for i in range(self._num_bands):
            if compute_bool[i] is True:
                kwargs_source_i = [kwargs_source[k] for k in self._index_source_list[i]]
                kwargs_lens_light_i = [kwargs_lens_light[k] for k in self._index_lens_light_list[i]]
                num += self._imageModel_list[i].num_param_linear(kwargs_lens, kwargs_source_i, kwargs_lens_light_i, kwargs_ps)
        return num