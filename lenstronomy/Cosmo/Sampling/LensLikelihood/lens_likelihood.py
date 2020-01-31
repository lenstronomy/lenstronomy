__author__ = 'sibirrer'

from lenstronomy.Cosmo.kde_likelihood import KDELikelihood
from lenstronomy.Cosmo.Sampling.LensLikelihood.hierarchical_cosmography import HierarchicalCosmography


class LensLikelihood(HierarchicalCosmography):
    """
    master class containing the likelihood definitions of different analysis
    """
    def __init__(self, z_lens, z_source, likelihood_type='TDKin', ani_param_array=None, ani_scaling_array=None,
                 **kwargs_likelihood):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param likelihood_type: string to specify the likelihood type
        :param ani_param_array: array of anisotropy parameter values for which the kinematics are predicted
        :param ani_scaling_array: velocity dispersion sigma**2 scaling of anisotropy parameter relative to default prediction
        :param kwargs_likelihood: keyword arguments specifying the likelihood function,
        see individual classes for their use
        """
        self._z_lens = z_lens
        self._z_source = z_source
        super(LensLikelihood, self).__init__(z_lens=z_lens, z_source=z_source, ani_param_array=ani_param_array,
                                             ani_scaling_array=ani_scaling_array)
        if likelihood_type == 'TDKin':
            self._lens_type = TDKinLikelihood(z_lens, z_source, **kwargs_likelihood)
        elif likelihood_type == 'Kin':
            self._lens_type = KinLikelihood(z_lens, z_source, **kwargs_likelihood)
        else:
            raise ValueError('likelihood_type %s not supported!' % likelihood_type)

    def lens_log_likelihood(self, cosmo, gamma_ppn=1, lambda_mst=1, kappa_ext=0, aniso_param=None):
        """

        :param cosmo: astropy.cosmology instance
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
        lambda_mst=1 corresponds to the input model
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :param aniso_param: global stellar anisotropy parameter
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
        ddt_, dd_ = self._displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst, kappa_ext=kappa_ext,
                                              aniso_param=aniso_param)
        return self._lens_type.log_likelihood(ddt_, dd_)


class TDKinLikelihood(object):
    """
    class for evaluating the 2-d posterior of Ddt vs Dd coming from a lens with time delays and kinematics measurement
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
        self._kde_likelihood = KDELikelihood(D_d_sample, D_delta_t_sample, kde_type=kde_type, bandwidth=bandwidth)

    def log_likelihood(self, ddt, dd):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        return self._kde_likelihood.logLikelihood(dd, ddt)


class KinLikelihood(object):
    """
    class to handle cosmographic likelihood coming from modeling lenses with imaging and kinematic data but no time delays.
    Thus Ddt is not constraint but the kinematics can constrain Ds/Dds

    The current version includes a Gaussian in Ds/Dds but can be extended.
    """
    def __init__(self, z_lens, z_source, ds_dds_mean, ds_dds_sigma):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ds_dds_mean: mean of Ds/Dds distance ratio
        :param ds_dds_sigma: 1-sigma uncertainty in the Ds/Dds distance ratio
        """
        self._z_lens = z_lens
        self._ds_dds_mean = ds_dds_mean
        self._ds_dds_sigma2 = ds_dds_sigma ** 2

    def log_likelihood(self, ddt, dd):
        """
        Note: kinematics + imaging data can constrain Ds/Dds. The input of Ddt, Dd is transformed here to match Ds/Dds

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :return: log likelihood given the single lens analysis
        """
        ds_dds = ddt / dd / (1 + self._z_lens)
        return - (ds_dds - self._ds_dds_mean) ** 2 / self._ds_dds_sigma2 / 2
