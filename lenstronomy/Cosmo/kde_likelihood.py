import numpy as np
from scipy import stats


class KDELikelihood(object):
    """
    class that samples the cosmographic likelihood given a distribution of points in the 2-dimensional distribution
    of D_d and D_delta_t
    """
    def __init__(self, D_d_sample, D_delta_t_sample, kde_type='scipy_gaussian', bandwidth=1):
        """

        :param D_d_sample: 1-d numpy array of angular diamter distances to the lens plane
        :param D_delta_t_sample: 1-d numpy array of time-delay distances
        kde_type : string
            The kernel to use.  Valid kernels are
            'scipy_gaussian' or
            ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
            Default is 'gaussian'.
        :param bandwidth: width of kernel (in same units as the angular diameter quantities
        """
        values = np.vstack([D_d_sample, D_delta_t_sample])
        if kde_type == 'scipy_gaussian':
            self._PDF_kernel = stats.gaussian_kde(values)
        else:
            from sklearn.neighbors import KernelDensity
            self._kde = KernelDensity(bandwidth=bandwidth, kernel=kde_type)
            values = np.vstack([D_d_sample, D_delta_t_sample])
            self._kde.fit(values.T)
        self._kde_type = kde_type

    def logLikelihood(self, D_d, D_delta_t):
        """
        likelihood of the data (represented in the distribution of this class) given a model with predicted angular
        diameter distances.

        :param D_d: model predicted angular diameter distance
        :param D_delta_t: model predicted time-delay distance
        :return: loglikelihood (log of KDE value)
        """
        if self._kde_type == 'scipy_gaussian':
            density = self._PDF_kernel([D_d, D_delta_t])
            logL = np.log(density)
        else:
            x = np.array([[D_d], [D_delta_t]])
            logL = self._kde.score_samples(x.T)
        return logL
