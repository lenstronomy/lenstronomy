__author__ = ['TheoDuboscq']

from lenstronomy.LensModel.LineOfSightFlexion.LOSFlexionModels.losflexion import LOSFlexion

__all__ = ['LOSFlexionMinimal']


class LOSFlexionMinimal(LOSFlexion):
    """
    Class deriving from LOSFlexion containing the parameters for line-of-sight corrections within 
    the "minimal model" defined in "Weak lensing of strong lensing: beyond the tidal regime" (in prep.). 
    It is equivalent to LOSFlexion but with fewer parameters (19), namely:
    kappa_od, gamma1_od, gamma2_od, F1_od, F2_od, G1_od, G2_od, kappa_los, gamma1_los, gamma2_los,
    F1_los, F2_los, G1_los, G2_los, F1_1los, F2_1los, G1_1los, G2_1los, omega_los.
    """

    param_names = ['kappa_od', 'gamma1_od', 'gamma2_od', 'F1_od', 'F2_od', 'G1_od', 'G2_od',
                   'kappa_los', 'gamma1_los', 'gamma2_los', 'F1_los', 'F2_los', 'G1_los', 'G2_los',
                   'F1_1los', 'F2_1los', 'G1_1los', 'G2_1los', 'omega_los']
    lower_limit_default = {pert: -0.5 for pert in param_names}
    upper_limit_default = {pert: 0.5 for pert in param_names}
