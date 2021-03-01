import numpy as np
from numpy.linalg import inv

__all__ = ['PositionLikelihood']


class PositionLikelihood(object):
    """
    likelihood of positions of multiply imaged point sources
    """
    def __init__(self, point_source_class, image_position_uncertainty=0.005, astrometric_likelihood=False,
                 image_position_likelihood=False, ra_image_list=[], dec_image_list=[],
                 source_position_likelihood=False, check_matched_source_position=False, source_position_tolerance=0.001,
                 source_position_sigma=0.001, force_no_add_image=False, restrict_image_number=False, max_num_images=None):
        """

        :param point_source_class: Instance of PointSource() class
        :param image_position_uncertainty: uncertainty in image position uncertainty (1-sigma Gaussian radially),
        this is applicable for astrometric uncertainties as well as if image positions are provided as data
        :param astrometric_likelihood: bool, if True, evaluates the astrometric uncertainty of the predicted and modeled
        image positions with an offset 'delta_x_image' and 'delta_y_image'
        :param image_position_likelihood: bool, if True, evaluates the likelihood of the model predicted image position given the data/measured image positions
        :param ra_image_list: list or RA image positions per model component
        :param dec_image_list: list or DEC image positions per model component
        :param source_position_likelihood: bool, if True, ray-traces image positions back to source plane and evaluates
        relative errors in respect ot the position_uncertainties in the image plane
        :param check_matched_source_position: bool, if True, checks whether multiple images are a solution of the same source
        :param source_position_tolerance: tolerance level (in arc seconds in the source plane) of the different images
        :param source_position_sigma: r.m.s. value corresponding to a 1-sigma Gaussian likelihood accepted by the model precision in matching the source position
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
        self._image_position_sigma = image_position_uncertainty
        self._source_position_sigma = source_position_sigma
        self._check_matched_source_position = check_matched_source_position
        self._bound_source_position_scatter = source_position_tolerance
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
            logL_astrometry = self.astrometric_likelihood(kwargs_ps, kwargs_special, self._image_position_sigma)
            logL += logL_astrometry
            if verbose is True:
                print('Astrometric likelihood = %s' % logL_astrometry)
        if self._check_matched_source_position is True:
            logL_source_scatter = self.source_position_likelihood(kwargs_lens, kwargs_ps, self._source_position_sigma, hard_bound_rms=self._bound_source_position_scatter, verbose=verbose)
            logL += logL_source_scatter
            if verbose is True:
                print('Source scatter punishing likelihood = %s' % logL_source_scatter)
        if self._force_no_add_image:
            bool = self.check_additional_images(kwargs_ps, kwargs_lens)
            if bool is True:
                logL -= 10.**5
                if verbose is True:
                    print('force no additional image penalty as additional images are found!')
        if self._restrict_number_images is True:
            ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
            if len(ra_image_list[0]) > self._max_num_images:
                logL -= 10.**5
                if verbose is True:
                    print('Number of images found %s exceeded the limited number allowed %s' % (len(ra_image_list[0]), self._max_num_images))
        if self._source_position_likelihood is True:
            logL_source_pos = self.source_position_likelihood(kwargs_lens, kwargs_ps, sigma=self._image_position_sigma)
            logL += logL_source_pos
            if verbose is True:
                print('source position likelihood %s' % logL_source_pos)
        if self._image_position_likelihood is True:
            logL_image_pos = self.image_position_likelihood(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens, sigma=self._image_position_sigma)
            logL += logL_image_pos
            if verbose is True:
                print('image position likelihood %s' % logL_image_pos)
        return logL

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

    @staticmethod
    def astrometric_likelihood(kwargs_ps, kwargs_special, sigma):
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
                return -np.inf
            return logL
        else:
            return 0

    def image_position_likelihood(self, kwargs_ps, kwargs_lens, sigma):
        """
        computes the likelihood of the model predicted image position relative to measured image positions with an astrometric error.
        This routine requires the 'ra_image_list' and 'dec_image_list' being declared in the initiation of the class

        :param kwargs_ps: point source keyword argument list
        :param kwargs_lens: lens model keyword argument list
        :param sigma: 1-sigma uncertainty in the measured position of the images
        :return: log likelihood of the model predicted image positions given the data/measured image positions.
        """
        ra_image_list, dec_image_list = self._pointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        logL = 0
        for i in range(len(ra_image_list)):  # sum over the images of the different model components
            logL += -np.sum(((ra_image_list[i] - self._ra_image_list[i])**2 + (dec_image_list[i] - self._dec_image_list[i])**2) / sigma**2 / 2)
        return logL

    def source_position_likelihood(self, kwargs_lens, kwargs_ps, sigma, hard_bound_rms=None, verbose=False):
        """
        computes a likelihood/punishing factor of how well the source positions of multiple images match given the image
        position and a lens model.
        The likelihood level is computed in respect of a displacement in the image plane and transposed through the
        Hessian into the source plane.

        :param kwargs_lens: lens model keyword argument list
        :param kwargs_ps: point source keyword argument list
        :param sigma: 1-sigma Gaussian uncertainty in the image plane
        :param hard_bound_rms: hard bound deviation between the mapping of the images back to the source plane (in source frame)
        :param verbose: bool, if True provides print statements with useful information.
        :return: log likelihood of the model reproducing the correct image positions given an image position uncertainty
        """
        if len(kwargs_ps) < 1:
            return 0
        logL = 0
        source_x, source_y = self._pointSource.source_position(kwargs_ps, kwargs_lens)
        for k in range(len(kwargs_ps)):
            if 'ra_image' in kwargs_ps[k] and self._pointSource.point_source_type_list[k] == 'LENSED_POSITION':

                x_image = kwargs_ps[k]['ra_image']
                y_image = kwargs_ps[k]['dec_image']
                # calculating the individual source positions from the image positions
                x_source, y_source = self._lensModel.ray_shooting(x_image, y_image, kwargs_lens)
                for i in range(len(x_image)):
                    f_xx, f_xy, f_yx, f_yy = self._lensModel.hessian(x_image[i], y_image[i], kwargs_lens)
                    A = np.array([[1 - f_xx, -f_xy], [-f_yx, 1 - f_yy]])
                    Sigma_theta = np.array([[1, 0], [0, 1]]) * sigma ** 2
                    Sigma_beta = image2source_covariance(A, Sigma_theta)
                    delta = np.array([source_x[k] - x_source[i], source_y[k] - y_source[i]])
                    if hard_bound_rms is not None:
                        if delta[0]**2 + delta[1]**2 > hard_bound_rms**2:
                            if verbose is True:
                                print('Image positions do not match to the same source position to the required precision. '
                                      'Achieved: %s, Required: %s.' % (delta, hard_bound_rms))
                            logL -= 10 ** 3
                    try:
                        Sigma_inv = inv(Sigma_beta)
                    except:
                        return -np.inf
                    chi2 = delta.T.dot(Sigma_inv.dot(delta))
                    logL -= chi2 / 2
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
