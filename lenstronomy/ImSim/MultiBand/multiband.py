from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.MultiBand.multi_data_base import MultiDataBase


class MultiBand(MultiDataBase):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    joint non-linear parameters and decoupled linear parameters.
    """

    def __init__(self, multi_band_list, lens_model_class=None, source_model_class=None, lens_light_model_class=None,
                 point_source_class=None):
        imageModel_list = []
        for i in range(len(multi_band_list)):
            kwargs_data = multi_band_list[i][0]
            kwargs_psf = multi_band_list[i][1]
            kwargs_numerics = multi_band_list[i][2]
            data_i = Data(kwargs_data=kwargs_data)
            psf_i = PSF(kwargs_psf=kwargs_psf)
            imageModel = ImageModel(data_i, psf_i, lens_model_class, source_model_class,
                                                    lens_light_model_class, point_source_class,
                                                    kwargs_numerics=kwargs_numerics)
            imageModel_list.append(imageModel)
        super(MultiBand, self).__init__(imageModel_list)

    def image_linear_solve(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, inv_bool=False):
        """
        computes the image (lens and source surface brightness with a given lens model).
        The linear parameters are computed with a weighted linear least square optimization (i.e. flux normalization of the brightness profiles)
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param inv_bool: if True, invert the full linear solver Matrix Ax = y for the purpose of the covariance matrix.
        :return: 1d array of surface brightness pixels of the optimal solution of the linear parameters to match the data
        """
        wls_list, error_map_list, cov_param_list, param_list = [], [], [], []
        for i in range(self._num_bands):
            wls_model, error_map, cov_param, param = self._imageModel_list[i].image_linear_solve(kwargs_lens,
                                                                                                 kwargs_source,
                                                                                                 kwargs_lens_light,
                                                                                                 kwargs_else,
                                                                                                 inv_bool=inv_bool)
            wls_list.append(wls_model)
            error_map_list.append(error_map)
            cov_param_list.append(cov_param)
            param_list.append(param)
        return wls_list, error_map_list, cov_param_list, param_list

    def image_positions(self, kwargs_ps, kwargs_lens):
        """
        lens equation solver for image positions given lens model and source position
        :param kwargs_lens: keyword arguments of lens models (as list)
        :param sourcePos_x: source position in relative arc sec
        :param sourcePos_y: source position in relative arc sec
        :return: x_coords, y_coords of image positions
        """
        x_mins, y_mins = self._imageModel_list[0].image_positions(kwargs_ps, kwargs_lens)
        return x_mins, y_mins

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
                logL += self._imageModel_list[i].likelihood_data_given_model(kwargs_lens, kwargs_source,
                                                                             kwargs_lens_light, kwargs_ps,
                                                                             source_marg=source_marg)
        return logL


