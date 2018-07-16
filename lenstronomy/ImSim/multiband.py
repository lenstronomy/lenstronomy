from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF


class Multiband(object):
    """
    class to simulate/reconstruct images in multi-band option.
    This class calls functions of image_model.py with different bands with
    joint non-linear parameters and decoupled linear parameters.
    """

    def __init__(self, multi_band_list, lens_model_class, source_model_class, lens_light_model_class, point_source_class):
        self._num_bands = len(multi_band_list)
        self.lensModel = lens_model_class
        self.pointSource = point_source_class
        self.sourceModel = source_model_class
        self.lensLightModel = lens_light_model_class
        self._imageModel_list = []
        for i in range(self._num_bands):
            kwargs_data = multi_band_list[i][0]
            kwargs_psf = multi_band_list[i][1]
            kwargs_numerics = multi_band_list[i][2]
            data_i = Data(kwargs_data=kwargs_data)
            psf_i = PSF(kwargs_psf=kwargs_psf)
            self._imageModel_list.append(ImageModel(data_i, psf_i, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics))

    def reset_point_source_cache(self):
        """
        deletes all the cache in the point source class and saves it from then on

        :return:
        """
        if self.pointSource is not None:
            self.pointSource.delete_lens_model_cach()
            self.pointSource.set_save_cache(True)
            for imageModel in self._imageModel_list:
                imageModel.PointSource.delete_lens_model_cach()
                imageModel.PointSource.set_save_cache(True)

    def source_surface_brightness(self, kwargs_source, kwargs_lens, unconvolved=False, de_lensed=False):
        """
        computes the source surface brightness distribution
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :return: list of 1d arrays of surface brightness pixels (for each band)
        """
        source_light_final_list = []
        for i in range(self._num_bands):
            source_light_final = self._imageModel_list[i].source_surface_brightness(kwargs_source, kwargs_lens, unconvolved=unconvolved, de_lensed=de_lensed)
            source_light_final_list.append(source_light_final)
        return source_light_final_list

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False):
        """
        computes the lens surface brightness distribution
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect seeing), otherwise convolved with PSF kernel
        :return: list of 1d array of surface brightness pixels (for each band)
        """
        lens_light_final_list = []
        for i in range(self._num_bands):
            lens_light_final = self._imageModel_list[i].lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
            lens_light_final_list.append(lens_light_final)
        return lens_light_final_list

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
            wls_model, error_map, cov_param, param = self._imageModel_list[i].image_linear_solve(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, inv_bool=inv_bool)
            wls_list.append(wls_model)
            error_map_list.append(error_map)
            cov_param_list.append(cov_param)
            param_list.append(param)
        return wls_list, error_map_list, cov_param_list, param_list

    def image(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True):
        """
        make a image with a realisation of linear parameter values "param"
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different lens light surface brightness profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 1d array of surface brightness pixels of the simulation
        """
        image_list = []
        for i in range(self._num_bands):
            image = self._imageModel_list[i].image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, unconvolved=unconvolved, source_add=source_add, lens_light_add=lens_light_add, point_source_add=point_source_add)
            image_list.append(image)
        return image_list

    def error_map(self, kwargs_lens, kwargs_ps):
        """

        :param kwargs_lens:
        :param kwargs_ps:
        :return:
        """
        error_map_list = []
        for i in range(self._num_bands):
            error_map_i = self._imageModel_list[i].error_map(kwargs_lens, kwargs_ps)
            error_map_list.append(error_map_i)
        return error_map_list

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

    def likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, source_marg=False, compute_bool=None):
        """
        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
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
                logL += self._imageModel_list[i].likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, source_marg=source_marg)
        return logL

    def numData_evaluate(self, compute_bool=None):
        if compute_bool is None:
            compute_bool = [True] * self._num_bands
        else:
            if not len(compute_bool) == self._num_bands:
                raise ValueError('compute_bool statement has not the same range as number of bands available!')
        num = 0
        for i in range(self._num_bands):
            if compute_bool[i] is True:
                num += self._imageModel_list[i].numData_evaluate
        return num

    def fermat_potential(self, kwargs_lens, kwargs_else):
        """

        :return: time delay in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """
        return self._imageModel_list[0].fermat_potential(kwargs_lens, kwargs_else)