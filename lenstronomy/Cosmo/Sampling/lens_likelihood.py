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
        dd = cosmo.angular_diameter_distance(z=self._z_lens).value
        ds = cosmo.angular_diameter_distance(z=self._z_source).value
        dds = cosmo.angular_diameter_distance_z1z2(z1=self._z_lens, z2=self._z_source).value
        ddt = (1. + self._z_lens) * dd * ds / dds

        # here we effectively change the posteriors of the lens, but rather than changing the instance of the KDE we
        # displace the predicted angular diameter distances in the opposite direction
        ddt_, dd_ = self._displace_prediction(ddt, dd, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst, kappa_ext=kappa_ext)
        return self._kde_likelihood.logLikelihood(dd_, ddt_)

    def _displace_prediction(self, ddt, dd, gamma_ppn=1, lambda_mst=1, kappa_ext=0):
        """
        here we effectively change the posteriors of the lens, but rather than changing the instance of the KDE we
        displace the predicted angular diameter distances in the opposite direction
        The displacements form different effects are multiplicative and thus invariant under the order those
        displacements are applied.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
        lambda_mst=1 corresponds to the input model
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: ddt_, dd_
        """
        ddt_, dd_ = self._displace_ppn(ddt, dd, gamma_ppn=gamma_ppn)
        ddt_, dd_ = self._displace_kappa_ext(ddt_, dd_, kappa_ext=kappa_ext)
        ddt_, dd_ = self._displace_lambda_mst(ddt_, dd_, lambda_mst=lambda_mst)
        return ddt_, dd_

    def _displace_ppn(self, ddt, dd, gamma_ppn=1):
        """
        post-Newtonian parameter sampling. The deflection terms remain the same as those are measured by lensing.
        The dynamical term changes and affects the kinematic prediction and thus dd

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :return: ddt_, dd_
        """
        dd_ = dd * (1 + gamma_ppn) / 2.
        return ddt, dd_

    def _displace_kappa_ext(self, ddt, dd, kappa_ext=0):
        """
        assumes an additional mass-sheet of kappa_ext is present at the lens LOS (effectively mimicing an overall
        selection bias in the lenses that is not visible in the individual LOS analyses of the lenses.
        This is speculative and should only be considered if there are specific reasons why the current LOS analysis
        is insufficient.

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: ddt_, dd_
        """
        ddt_ = ddt / (1. - kappa_ext)
        return ddt_, dd

    def _displace_lambda_mst(self, ddt, dd, lambda_mst=1):
        """
        approximate internal mass-sheet transform on top of the assumed profiles inferred in the analysis of the
        individual lenses. The effect is to first order the same as for a pure mass sheet as a kappa_ext term.
        However the change here affects the 3-dimensional mass profile and thus the kinematics predictions is affected.
        We showed that for a set of profiles, the kinematics of a 3-d approximate mass sheet can still be very well
        approximated as d sigma_v_lambda**2 /d lambda = lambda * sigma_v0

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
        lambda_mst=1 corresponds to the input model, 0.9 corresponds to a positive mass sheet of 0.1
        kappa_ext = 1 - lambda_mst
        :return: ddt_, dd_
        """
        ddt_ = ddt * lambda_mst  # the actual posteriors needed to be corrected by Ddt_true = Ddt_mst / (1-kappa_ext)
        # this line can be changed in case the physical 3-d approximation of the chosen profile does scale differently with the kinematics
        sigma_v2_scaling = lambda_mst
        dd_ = dd * sigma_v2_scaling / lambda_mst  # the kinematics constrain Dd/Dds and thus the constraints on Dd needs to devide out the change in Ddt
        return ddt_, dd_


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

    def log_likelihood(self, cosmo, gamma_ppn=1, lambda_mst=1, kappa_ext=0):
        """

        :param cosmo: astropy.cosmology instance
        :param gamma_ppn: post-newtonian gravity parameter (=1 is GR)
        :param lambda_mst: overall global mass-sheet transform applied on the sample,
        lambda_mst=1 corresponds to the input model
        :param kappa_ext: external convergence to be added on top of the D_dt posterior
        :return: log likelihood of the combined lenses
        """
        logL = 0
        for lens in self._lens_list:
            logL += lens.lens_log_likelihood(cosmo=cosmo, gamma_ppn=gamma_ppn, lambda_mst=lambda_mst, kappa_ext=kappa_ext)
        return logL
