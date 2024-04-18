__author__ = 'TheoDuboscq'

__all__ = ['LOSFlexion']


class LOSFlexion(object):
    """
    Class allowing one to add line-of-sight effects up to flexion (convergence, shear and type-F and G flexion) to single-plane
    lensing. As the LOS class, this is not a profile, but when present in list of lens models, it is automatically recognised by
    ModelAPI(), which sets the flag los_flexion_effects to True, and thereby leads LensModel to use SinglePlaneLOSFlexion()
    instead of SinglePlane(). It is however incompatible with MultiPlane() just like the LOS class.

    The key-word arguments are the three line-of-sight convergences, the two components of the three line-of-sight shears,
    the two components of the four line-of-sight type-F flexion, and the two components of the four line-of-sight type-G 
    flexion, for a total of 25 real numbers. Those are named as in "Weak lensing of strong lensing: beyond the tidal regime" (in prep.) :
    kappa_od, kappa_os, kappa_ds, gamma1_od, gamma2_od, gamma1_os, gamma2_os, gamma1_ds, gamma2_ds, F1_od, F2_od, G1_od, G2_od,
    F1_os, F2_os, G1_os, G2_os, F1_1ds, F2_1ds, G1_1ds, G2_1ds, F1_2ds, F2_2ds, G1_2ds, G2_2ds. On top of this is added the rotation
    omega_os, which will serve in the minimal model where it transforms into the non zero omega_los.

    Because LOSFlexion is not a profile, it does not contain the usual functions function(), derivatives(), and hessian(),
    but rather modifies the behaviour of those functions in the SinglePlaneLOS() class.

    Instead, it contains the essential building blocks of this modification.
    """

    param_names = ['kappa_od', 'kappa_os', 'kappa_ds',
                   'gamma1_od', 'gamma2_od',
                   'gamma1_os', 'gamma2_os',
                   'gamma1_ds', 'gamma2_ds', 
                   'F1_od', 'F2_od', 'G1_od', 'G2_od',
                   'F1_os', 'F2_os', 'G1_os', 'G2_os', 
                   'F1_1ds', 'F2_1ds', 'G1_1ds', 'G2_1ds', 
                   'F1_2ds', 'F2_2ds', 'G1_2ds', 'G2_2ds', 'omega_os'] 
    lower_limit_default = {pert: -0.5 for pert in param_names}
    upper_limit_default = {pert: 0.5 for pert in param_names}


    def __init__(self, *args, **kwargs):
        self._static = False
        

    def set_static(self, **kwargs):
        """
        pre-computes certain computations that do only relate to the lens model parameters and not to the specific
        position where to evaluate the lens model

        :param kwargs: lens model parameters
        :return: no return, for certain lens model some private self variables are initiated
        """
        pass

    def set_dynamic(self):
        """

        :return: no return, deletes pre-computed variables for certain lens models
        """
        pass
