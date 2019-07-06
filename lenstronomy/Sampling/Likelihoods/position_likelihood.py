import numpy as np


class PositionLikelihood(object):
    """
    likelihood of positions of multiply imaged point sources
    """
    def __init__(self, point_source_class, param_class, astrometric_likelihood, position_uncertainty, check_solver,
                 solver_tolerance, force_no_add_image, restrict_image_number, max_num_images):
        """

        :param point_source_class: Instance of PointSource() class
        :param param_class: Instance of Param() class
        :param astrometric_likelihood: bool, if True, evaluates the astrometric uncertainty of the predicted and modeled
        image positions
        :param position_uncertainty: uncertainty in image position uncertainty (1-sigma Gaussian)
        :param check_solver: bool, if True, checks whether multiple images are a solution of the same source
        :param solver_tolerance: tolerance level (in arc seconds in the source plane) of the different images
        :param force_no_add_image: bool, if True, will punish additional images appearing in the frame of the modelled
        image(first calculate them)
        :param restrict_image_number: bool, if True, searches for all appearing images in the frame of the data and
        compares with max_num_images
        :param max_num_images: integer, maximum number of appearing images. Default is the number of  images given in
        the Param() class
        """
        self._pointSource = point_source_class
        self._param = param_class
        self._astrometric_likelihood = astrometric_likelihood
        self._position_sigma = position_uncertainty
        self._check_solver = check_solver
        self._solver_tolerance = solver_tolerance
        self._force_no_add_image = force_no_add_image
        self._restrict_number_images = restrict_image_number
        if max_num_images is None:
            max_num_images = self._param.num_point_source_images
        self._max_num_images = max_num_images

    def logL(self, kwargs_lens, kwargs_ps, kwargs_cosmo, verbose=False):

        logL = 0
        if self._astrometric_likelihood is True:
            logL_astrometry = self.astrometric_likelihood(kwargs_ps, kwargs_cosmo, self._position_sigma)
            logL += logL_astrometry
            if verbose is True:
                print('Astrometric likelihood = %s' % logL_astrometry)
        if self._check_solver is True:
            logL -= self.solver_penalty(kwargs_lens, kwargs_ps, kwargs_cosmo, self._solver_tolerance, verbose=verbose)
        if self._force_no_add_image:
            bool = self.check_additional_images(kwargs_ps, kwargs_lens)
            if bool is True:
                logL -= 10**10
                if verbose is True:
                    print('force no additional image penalty as additional images are found!')
        if self._restrict_number_images is True:
            ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
            if len(ra_image_list[0]) > self._max_num_images:
                logL -= 10**10
                if verbose is True:
                    print('Number of images found %s exceeded the limited number allowed %s' % (len(ra_image_list[0]), self._max_num_images))
        return logL

    def solver_penalty(self, kwargs_lens, kwargs_ps, kwargs_cosmo, tolerance, verbose=False):
        """
        test whether the image positions map back to the same source position
        :param kwargs_lens:
        :param kwargs_ps:
        :return: add penalty when solver does not find a solution
        """
        dist = self._param.check_solver(kwargs_lens, kwargs_ps, kwargs_cosmo)
        if dist > tolerance:
            if verbose is True:
                print('Image positions do not match to the same source position to the required precision. '
                      'Achieved: %s, Required: %s.' % (dist, tolerance))
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

    def astrometric_likelihood(self, kwargs_ps, kwargs_cosmo, sigma):
        """
        evaluates the astrometric uncertainty of the model plotted point sources (only available for 'LENSED_POSITION'
        point source model) and predicted image position by the lens model including an astrometric correction term.

        :param kwargs_ps: point source model kwargs list
        :param kwargs_cosmo: kwargs list, should include the astrometric corrections 'delta_x', 'delta_y'
        :param sigma: 1-sigma Gaussian uncertainty in the astrometry
        :return: log likelihood of the astrometirc correction between predicted image positions and model placement of the point sources
        """
        if 'ra_image' not in kwargs_ps[0]:
            return 0
        x_image = kwargs_ps[0]['ra_image']
        y_image = kwargs_ps[0]['dec_image']
        x_pos, y_pos = self._param.real_image_positions(x_image, y_image, kwargs_cosmo)
        dist = ((x_pos - x_image)**2 + (y_pos - y_image)**2)/sigma**2/2
        logL = -np.sum(dist)
        if np.isnan(logL) is True:
            return -10 ** 15
        return logL
