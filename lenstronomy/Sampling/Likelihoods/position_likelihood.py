import numpy as np
from numpy.linalg import inv


class PositionLikelihood(object):
    """
    likelihood of positions of multiply imaged point sources
    """
    def __init__(self, point_source_class, position_uncertainty=0.005, astrometric_likelihood=False,
                 image_position_likelihood=False, ra_image_list=[], dec_image_list=[],
                 source_position_likelihood=False, check_solver=False, solver_tolerance=0.001, force_no_add_image=False,
                 restrict_image_number=False, max_num_images=None):
        """

        :param point_source_class: Instance of PointSource() class
        :param position_uncertainty: uncertainty in image position uncertainty (1-sigma Gaussian),
        this is applicable for astrometric uncertainties as well as if image positions are provided as data
        :param astrometric_likelihood: bool, if True, evaluates the astrometric uncertainty of the predicted and modeled
        image positions with an offset 'delta_x_image' and 'delta_y_image'
        :param image_position_likelihood: bool, if True, evaluates the likelihood of the model predicted image position given the data/measured image positions
        :param ra_image_list: list or RA image positions per model component
        :param dec_image_list: list or DEC image positions per model component
        :param source_position_likelihood: bool, if True, ray-traces image positions back to source plane and evaluates
        relative errors in respect ot the position_uncertainties in the image plane
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
        # TODO replace with public function of ray_shooting
        self._lensModel = point_source_class._lensModel
        self._astrometric_likelihood = astrometric_likelihood
        self._position_sigma = position_uncertainty
        self._check_solver = check_solver
        self._solver_tolerance = solver_tolerance
        self._force_no_add_image = force_no_add_image
        self._restrict_number_images = restrict_image_number
        self._source_position_likelihood = source_position_likelihood
        self._max_num_images = max_num_images
        if max_num_images is None and restrict_image_number is True:
            raise ValueError('max_num_images needs to be provided when restrict_number_images is True!')
        self._image_position_likelihood = image_position_likelihood
        self._ra_image_list, self._dec_image_list = ra_image_list, dec_image_list

    def logL(self, kwargs_lens, kwargs_ps, kwargs_special, verbose=False):
        """

        :param kwargs_lens: lens model parameter keyword argument list
        :param kwargs_ps: point source model parameter keyword argument list
        :param kwargs_special: special keyword arguments
        :param verbose: bool
        :return: log likelihood of the optional likelihoods being computed
        """

        logL = 0
        if self._astrometric_likelihood is True:
            logL_astrometry = self.astrometric_likelihood(kwargs_ps, kwargs_special, self._position_sigma)
            logL += logL_astrometry
            if verbose is True:
                print('Astrometric likelihood = %s' % logL_astrometry)
        if self._check_solver is True:
            logL -= self.solver_penalty(kwargs_lens, kwargs_ps, self._solver_tolerance, verbose=verbose)
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
        if self._source_position_likelihood is True:
            logL_source_pos = self.source_position_likelihood(kwargs_lens, kwargs_ps, sigma=self._position_sigma)
            logL += logL_source_pos
            if verbose is True:
                print('source position likelihood %s' % logL_source_pos)
        if self._image_position_likelihood is True:
            logL_image_pos = self.image_position_likelihood(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens, sigma=self._position_sigma)
            logL += logL_image_pos
            if verbose is True:
                print('image position likelihood %s' % logL_image_pos)
        return logL

    def solver_penalty(self, kwargs_lens, kwargs_ps, tolerance, verbose=False):
        """
        test whether the image positions map back to the same source position
        :param kwargs_lens:
        :param kwargs_ps:
        :return: add penalty when solver does not find a solution
        """
        if len(kwargs_ps) < 1:
            return 0
        if 'ra_image' in kwargs_ps[0]:
            ra_image, dec_image = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
            source_x, source_y = self._lensModel.ray_shooting(ra_image, dec_image, kwargs_lens)
            dist = np.sqrt(np.sum((source_x - source_x[0]) ** 2 + (source_y - source_y[0]) ** 2))
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
            if 'ra_image' in kwargs_ps[0]:
                if len(ra_image_list[0]) > len(kwargs_ps[0]['ra_image']):
                    return True
        return False

    def astrometric_likelihood(self, kwargs_ps, kwargs_special, sigma):
        """
        evaluates the astrometric uncertainty of the model plotted point sources (only available for 'LENSED_POSITION'
        point source model) and predicted image position by the lens model including an astrometric correction term.

        :param kwargs_ps: point source model kwargs list
        :param kwargs_special: kwargs list, should include the astrometric corrections 'delta_x', 'delta_y'
        :param sigma: 1-sigma Gaussian uncertainty in the astrometry
        :return: log likelihood of the astrometirc correction between predicted image positions and model placement of the point sources
        """
        if not len(kwargs_ps) > 0:
            return 0
        if 'ra_image' not in kwargs_ps[0]:
            return 0
        if 'delta_x_image' in kwargs_special:
            delta_x, delta_y = np.array(kwargs_special['delta_x_image']), np.array(kwargs_special['delta_y_image'])
            dist = (delta_x ** 2 + delta_y ** 2) / sigma ** 2 / 2
            logL = -np.sum(dist)
            if np.isnan(logL) is True:
                return -10 ** 15
            return logL
        else:
            return 0

    def image_position_likelihood(self, kwargs_ps, kwargs_lens, sigma):
        """

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :return: log likelihood of the model predicted image positions given the data/measured image positions.
        """
        ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        logL = 0
        for i in range(len(ra_image_list)):  # sum over the images of the different model components
            logL += -np.sum((ra_image_list[i] - self._ra_image_list[i])**2 / sigma**2 / 2)
        return logL

    def source_position_likelihood(self, kwargs_lens, kwargs_ps, sigma):
        """
        computes a likelihood/punishing factor of how well the source positions of multiple images match.
        The likelihood level is computed in respect of a displacement in the image plane and transposed through the
        Hessian into the source plane

        :param kwargs_lens:
        :param kwargs_ps:
        :return: log likelihood of the model reproducing the correct image positions given an image position uncertainty
        """
        if 'ra_image' not in kwargs_ps[0]:
            return 0
        logL = 0
        source_x, source_y = self._pointSource.source_position(kwargs_ps, kwargs_lens)

        x_image = kwargs_ps[0]['ra_image']
        y_image = kwargs_ps[0]['dec_image']
        # calculating the individual source positions from the image positions
        x_source, y_source = self._lensModel.ray_shooting(x_image, y_image, kwargs_lens)
        for i in range(len(x_image)):
            f_xx, f_xy, f_yx, f_yy = self._lensModel.hessian(x_image[i], y_image[i], kwargs_lens)
            A = np.array([[1 - f_xx, -f_xy], [-f_yx, 1 - f_yy]])
            Sigma_theta = np.array([[1, 0], [0, 1]]) * sigma ** 2
            Sigma_beta = image2source_covariance(A, Sigma_theta)
            delta = np.array([source_x - x_source[i], source_y - y_source[i]])
            try:
                Sigma_inv = inv(Sigma_beta)
            except:
                return -10**15
            chi2 = delta.T.dot(Sigma_inv.dot(delta))[0][0]
            logL -= chi2/2
        return logL

    @property
    def num_data(self):
        """

        :return: integer, number of data points assocated with the class instance
        """
        num = 0
        if self._image_position_likelihood is True:
            for i in range(len(self._ra_image_list)):  # sum over the images of the different model components
                num += len(self._ra_image_list[i]) * 2
        return num


# Equation (13) in Birrer & Treu 2019
def image2source_covariance(A, Sigma_theta):
    """
    computes error covariance in the source plane
    A: Hessian lensing matrix
    Sigma_theta: image plane covariance matrix of uncertainties
    """
    ATSigma = np.matmul(A.T, Sigma_theta)
    return np.matmul(ATSigma, A)
