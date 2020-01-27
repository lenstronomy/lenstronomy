__author__ = 'sibirrer'

from lenstronomy.Cosmo.kde_likelihood import KDELikelihood
from lenstronomy.Cosmo.Sampling.LensLikelihood.hierarchical_cosmography import HierarchicalCosmography


class LensLikelihood(HierarchicalCosmography):
    """
    master class containing the likelihood definitions of different analysis
    """
    def __init__(self, z_lens, z_source, likelihood_type='TDKin', kwargs_likelihood={}):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param likelihood_type: string to specify the likelihood type
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        self._z_lens = z_lens
        self._z_source = z_source
        super(LensLikelihood, self).__init__(z_lens=z_lens, z_source=z_source)
        if likelihood_type == 'TDKin':
            self._lens_type = TDKinLikelihood(**kwargs_likelihood)
        else:
            ValueError('likelihood_type %s not supported!' % likelihood_type)

    def lens_log_likelihood(self, cosmo, gamma_ppn=1, lambda_mst=1, kappa_ext=0):
        """

        :param cosmo: astropy.cosmology instance
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
        lambda_mst=1 corresponds to the input model
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: log likelihood of the data given the model
        """

        # here we compute the unperturbed angular diameter distances of the lens system given the cosmology
        # Note: Distances are in physical units of Mpc. Make sure the posteriors to evaluate this likelihood is in the
        # same units
        dd = cosmo.angular_diameter_distance(z=self._z_lens).value
        ds = cosmo.angular_diameter_distance(z=self._z_source).value
        dds = cosmo.angular_diameter_distance_z1z2(z1=self._z_lens, z2=self._z_source).value
        ddt = (1. + self._z_lens) * dd * ds / dds

        # here we effectively change the posteriors of the lens, but rather than changing the instance of the KDE we
        # displace the predicted angular diameter distances in the opposite direction
        ddt_, dd_ = self._displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst, kappa_ext=kappa_ext)
        return self._lens_type.log_likelihood(ddt_, dd_)


class TDKinLikelihood(object):
    """
    class for evaluating the 2-d posterior of Ddt vs Dd coming from a lens with time delays and kinematics measurement
    """
    def __init__(self, D_d_sample, D_delta_t_sample, kde_type='scipy_gaussian', bandwidth=1):
        """

        :param D_d_sample: angular diameter to the lens posteriors (in physical Mpc)
        :param D_delta_t_sample: time-delay distance posteriors (in physical Mpc)
        :param kde_type: kernel density estimator type (see KDELikelihood class)
        :param bandwidth: width of kernel (in same units as the angular diameter quantities)
        """
        self._kde_likelihood = KDELikelihood(D_d_sample, D_delta_t_sample, kde_type=kde_type, bandwidth=bandwidth)

    def log_likelihood(self, ddt, dd):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return self._kde_likelihood.logLikelihood(dd, ddt)
