from lenstronomy.ImSim.make_image import MakeImage


class MakeImageMultiband(object):
    """
    class to simulate/reconstruct images in multiband option
    """

    def __init__(self, kwargs_options, kwargs_data_list=[], kwargs_psf_list=[]):
        self._num_bands = len(kwargs_data_list)
        if self._num_bands != len(kwargs_psf_list):
            raise ValueError("not equal number of PSF and Data configurations provided!")
        self._makeImage_list = []
        for i in range(self._num_bands):
            self._makeImage_list.append(MakeImage(kwargs_options, kwargs_data_list[i], kwargs_psf_list[i]))

        self.kwargs_options = kwargs_options

    def source_surface_brightness(self, kwargs_lens, kwargs_source, kwargs_else, unconvolved=False, de_lensed=False):
        """
        computes the source surface brightness distribution
        :param kwargs_lens: list of keyword arguments corresponding to the superposition of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the superposition of different source light profiles
        :param kwargs_else: keyword arguments corresponding to "other" parameters, such as external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness profile, otherwise the lensed.
        :return: list of 1d arrays of surface brightness pixels (for each band)
        """
        source_light_final_list = []
        for i in range(self._num_bands):
            source_light_final = self._makeImage_list[i].source_surface_brightness(kwargs_lens, kwargs_source, kwargs_else, unconvolved=unconvolved, de_lensed=de_lensed)
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
            lens_light_final = self._makeImage_list[i].lens_surface_brightness(kwargs_lens_light, unconvolved=unconvolved)
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
            wls_model, error_map, cov_param, param = self._makeImage_list[i].image_linear_solve(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, inv_bool=inv_bool)
            wls_list.append(wls_model)
            error_map_list.append(error_map)
            cov_param_list.append(cov_param)
            param_list.append(param)
        return wls_list, error_map_list, cov_param_list, param_list

    def image_with_params(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True):
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
        error_map_list = []
        for i in range(self._num_bands):
            image, error_map = self._makeImage_list[i].image_with_params(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, unconvolved=unconvolved, source_add=source_add, lens_light_add=lens_light_add, point_source_add=point_source_add)
            image_list.append(image)
            error_map_list.append(error_map)
        return image_list, error_map_list

    def image_positions(self, kwargs_lens, kwargs_else, sourcePos_x, sourcePos_y):
        """

        :param kwargs_lens:
        :param kwargs_else:
        :param sourcePos_x: source position in relative arc sec
        :param sourcePos_y: source position in relative arc sec
        :return:
        """
        x_mins, y_mins = self._makeImage_list[0].image_position(kwargs_lens, kwargs_else, sourcePos_x, sourcePos_y)
        return x_mins, y_mins

    def likelihood_data_given_model(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else):
        """
        computes the likelihood of the data given a model
        This is specified with the non-linear parameters and a linear inversion and prior marginalisation.
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :return: log likelihood (natural logarithm) (sum of the log likelihoods of the individual images)
        """
        # generate image
        logL = 0
        for i in range(self._num_bands):
            logL += self._makeImage_list[i].likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
        return logL