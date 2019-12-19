__author__ = 'sibirrer'

from lenstronomy.Cosmo.kde_likelihood import KDELikelihood


class LensLikelihood(object):
    """
    class for evaluating single lens likelihood
    """
    def __init__(self, z_lens, z_source, D_d_sample, D_delta_t_sample, kde_type='scipy_gaussian', bandwidth=1):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param D_d_sample: angular diameter to the lens posteriors (in physical Mpc)
        :param D_delta_t_sample: time-delay distance posteriors (in physical Mpc)
        :param kde_type: kernel density estimator type (see KDELikelihood class)
        :param bandwidth: width of kernel (in same units as the angular diameter quantities)
        """
        self._z_lens = z_lens
        self._z_source = z_source
        self._kde_likelihood = KDELikelihood(D_d_sample, D_delta_t_sample, kde_type=kde_type, bandwidth=bandwidth)

    def lens_log_likelihood(self, cosmo, gamma_ppn=1, kappa_ext=0):
        """

        :param cosmo: astropy.cosmology instance
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: log likelihood of the data given the model
        """
        dd = cosmo.angular_diameter_distance(z=self._z_lens).value
        ds = cosmo.angular_diameter_distance(z=self._z_source).value
        dds = cosmo.angular_diameter_distance_z1z2(z1=self._z_lens, z2=self._z_source).value
        ddt = (1. + self._z_lens) * dd * ds / dds
        dd_ = dd * 2. / (1 + gamma_ppn)
        ddt_ = ddt / (1 - kappa_ext)
        return self._kde_likelihood.logLikelihood(dd_, ddt_)


class LensSampleLikelihood(object):
    """
    class to evaluate the likelihood of a cosmology given a sample of angular diameter posteriors
    Currently this class does not include possible covariances between the lens samples
    """
    def __init__(self, kwargs_lens_list):
        """

        :param kwargs_lens_list: keyword argument list specifying the arguments of the LensLikelihood class
        """
        self._lens_list = []
        for kwargs_lens in kwargs_lens_list:
            self._lens_list.append(LensLikelihood(**kwargs_lens))

    def log_likelihood(self, cosmo, gamma_ppn=1, kappa_ext=0):
        """

        :param cosmo: astropy.cosmology instance
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: log likelihood of the combined lenses
        """
        logL = 0
        for lens in self._lens_list:
            logL += lens.lens_log_likelihood(cosmo=cosmo, gamma_ppn=gamma_ppn, kappa_ext=kappa_ext)
        return logL
