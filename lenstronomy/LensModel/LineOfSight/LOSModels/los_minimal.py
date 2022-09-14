__author__ = 'nataliehogg', 'pierrefleury'

from lenstronomy.LensModel.LineOfSight.LOSModels.los import LOS

__all__ = ['LOSMinimal']

class LOSMinimal(LOS):
    """
    Class deriving from LOS containing the parameters for line-of-sight
    corrections within the "minimal model" defined in
    https://arxiv.org/abs/2104.08883
    It is equivalent to LOS but with fewer parameters, namely:
    kappa_od, gamma1_od, gamma2_od, omega_od, kappa_los, gamma1_los,
    gamma2_los, omega_los.
    """

    param_names = ['kappa_od', 'gamma1_od','gamma2_od', 'omega_od',
                   'kappa_los', 'gamma1_los','gamma2_los', 'omega_los']
    lower_limit_default = {pert: -0.5 for pert in param_names}
    upper_limit_default = {pert: 0.5 for pert in param_names}
