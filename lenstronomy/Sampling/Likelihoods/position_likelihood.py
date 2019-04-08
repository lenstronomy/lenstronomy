import numpy as np


class PositionLikelihood(object):
    """
    likelihood of positions of multiply imaged point sources
    """
    def __init__(self, point_source_class, param_class, point_source_likelihood, position_uncertainty, check_solver, solver_tolerance, force_no_add_image,
                 restrict_image_number, max_num_images):
        """

        :param point_source_class:
        :param param_class:
        :param position_uncertainty:
        :param check_solver:
        :param solver_tolerance:
        :param force_no_add_image:
        :param restrict_image_number:
        :param max_num_images:
        """
        self._pointSource = point_source_class
        self._param = param_class
        self._point_source_likelihood = point_source_likelihood
        self._position_sigma = position_uncertainty
        self._check_solver = check_solver
        self._solver_tolerance = solver_tolerance
        self._force_no_add_image = force_no_add_image
        self._restrict_number_images = restrict_image_number
        if max_num_images is None:
            max_num_images = self._param.num_point_source_images
        self._max_num_images = max_num_images

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):

        logL = 0
        if self._point_source_likelihood is True:
            logL += self.likelihood_image_pos(kwargs_lens, kwargs_ps, kwargs_cosmo, self._position_sigma)

        if self._check_solver is True:
            logL -= self.solver_penalty(kwargs_lens, kwargs_ps, kwargs_cosmo, self._solver_tolerance)
        if self._force_no_add_image:
            bool = self.check_additional_images(kwargs_ps, kwargs_lens)
            if bool is True:
                logL -= 10**10
        if self._restrict_number_images is True:
            ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
            if len(ra_image_list[0]) > self._max_num_images:
                logL -= 10**10
        return logL

    def solver_penalty(self, kwargs_lens, kwargs_ps, kwargs_cosmo, tolerance):
        """
        test whether the image positions map back to the same source position
        :param kwargs_lens:
        :param kwargs_ps:
        :return: add penalty when solver does not find a solution
        """
        dist = self._param.check_solver(kwargs_lens, kwargs_ps, kwargs_cosmo)
        if dist > tolerance:
            return dist * 10**10
        return 0

    def check_additional_images(self, kwargs_ps, kwargs_lens):
        """
        checks whether additional images have been found and placed in kwargs_ps
        :param kwargs_ps: point source kwargs
        :return: bool, True if more image positions are found than originally been assigned
        """
        ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        if len(ra_image_list) > 0:
            if len(ra_image_list[0]) > self._param.num_point_source_images:
                return True
        return False

    def likelihood_image_pos(self, kwargs_lens, kwargs_ps, kwargs_cosmo, sigma):
        """

        :param x_lens_model: image position of lens model
        :param y_lens_model: image position of lens model
        :param x_image: image position of image data
        :param y_image: image position of image data
        :param sigma: likelihood sigma
        :return: log likelihood of model given image positions
        """
        if not 'ra_image' in kwargs_ps[0]:
            return 0
        x_image = kwargs_ps[0]['ra_image']
        y_image = kwargs_ps[0]['dec_image']
        ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        x_pos, y_pos = self._param.real_image_positions(ra_image_list[0], dec_image_list[0], kwargs_cosmo)
        num_image = len(ra_image_list[0])
        if num_image != len(x_image):
            return -10**15
        dist = ((x_pos - x_image)**2 + (y_pos - y_image)**2)/sigma**2/2
        logL = -np.sum(dist)
        if np.isnan(logL) is True:
            return -10 ** 15
        return logL


class FluxRatioLikelihood(object):
    """
    likelihood class for magnification of multiply lensed images
    """

    def __init__(self, point_source_class, lens_model_class, param_class, flux_ratios, flux_ratio_errors):
        """

        :param point_source_class: PointSource class instance
        :param lens_model_class: LensModel class instance
        :param param_class: Param() class instance
        :param flux_ratios: ratio of fluxes of the multiple images (relative to the first appearing)
        :param flux_ratio_errors: errors in the flux ratios (relative to the first appearing
        """
        self._pointSource = point_source_class
        self._lens_model_class = lens_model_class
        self._param = param_class
        self._flux_ratios = np.array(flux_ratios)
        self._flux_ratio_errors = np.array(flux_ratio_errors)

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo):
        """

        :param kwargs_lens:
        :param kwargs_ps:
        :param kwargs_cosmo:
        :return: log likelihood of the measured flux ratios given a model
        """

        ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        x_pos, y_pos = self._param.real_image_positions(ra_image_list[0], dec_image_list[0], kwargs_cosmo)
        mag = np.abs(self._lens_model_class.magnification(x_pos, y_pos, kwargs_lens))
        mag_ratio = mag[1:] - mag[0]
        dist = (mag_ratio - self._flux_ratios)**2 / self._flux_ratio_errors**2 / 2
        logL = -np.sum(dist)
        if np.isnan(logL) is True:
            return -10 ** 15
        return logL