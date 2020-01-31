__author__ = 'sibirrer'
from scipy.interpolate import interp1d


class HierarchicalCosmography(object):
    """
    class to manage hierarchical hyper-parameter that impact the cosmographic posterior interpretation of individual
    lenses.
    """

    def __init__(self, z_lens, z_source, ani_param_array=None, ani_scaling_array=None):
        """

        :param z_lens: lens redshift
        :param z_source: source redshift
        :param ani_param_array: array of anisotropy parameter values for which the kinematics are predicted
        :param ani_scaling_array: velocity dispersion sigma**2 scaling of anisotropy parameter relative to default prediction

        """
        self._z_lens = z_lens
        self._z_source = z_source
        if ani_param_array is not None and ani_param_array is not None:
            self._f_ani = interp1d(ani_param_array, ani_scaling_array, kind='cubic')

    def _displace_prediction(self, ddt, dd, gamma_ppn=1, lambda_mst=1, kappa_ext=0, aniso_param=None):
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
        :param aniso_param: global stellar anisotropy parameter
        :return: ddt_, dd_
        """
        ddt_, dd_ = self._displace_ppn(ddt, dd, gamma_ppn=gamma_ppn)
        ddt_, dd_ = self._displace_kappa_ext(ddt_, dd_, kappa_ext=kappa_ext)
        ddt_, dd_ = self._displace_lambda_mst(ddt_, dd_, lambda_mst=lambda_mst)
        ddt_, dd_ = self._displace_anisotropy(ddt_, dd_, anisotropy_param=aniso_param)
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

    def _displace_anisotropy(self, ddt, dd, anisotropy_param):
        """

        :param ddt: time-delay distance
        :param dd: angular diameter distance to the deflector
        :param anisotropy_param: anisotropy parameter that changes the predicted Ds/Dds from the kinematic by:
        Ds/Dds(aniso_param) = f(aniso_param) * Ds/Dds(initial)

        :return: inverse predicted offset in Ds/Dds by the anisotropy model deviating from the original sample
        """

        if anisotropy_param is None or not hasattr(self, '_f_ani'):
            dd_ = dd
        else:
            dd_ = dd * self._f_ani(anisotropy_param)
        return ddt, dd_
