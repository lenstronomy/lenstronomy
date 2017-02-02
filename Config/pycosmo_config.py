__author__ = 'sibirrer'

# Cosmology Input Parameters
#--------------------

h = 0.7
"""dimensionless Hubble constant [1]"""

omega_b = 0.06
"""Baryon density parameter [1]"""

omega_m = 0.3
"""Matter density paramater (dark matter + baryons) [1]"""

omega_l_in = "flat"
#omega_l_in = 0.7
"""Dark energy density. If 'flat' then omega_l is 1.- omega_m - omega_r [1]"""

w0 = -1.0
"""DE equation of state at z=0 [1]"""

wa = 0.0
"""DE equation of state evolution such that w(a)=w0+wa(1-a) [1]"""

n = 1.0
"""Spectral index for scalar modes [1]"""

tau = 0.09
"""Optical depth [under development]"""

#pk_norm_type='deltah'
pk_norm_type='sigma8'
"""Power spectrum normalisation scheme: 'deltah' for CMB normalisation or 'sigma8' for sigma8 normalisation"""

#pk_norm = 4.6*10**-5    # deltah example
pk_norm=.8 # sigma8 example
"""Power spectrum normalisation value: either deltah or sigma8 depending on pk_norm_type setting"""

Tcmb = 2.725
"""CMB temperature [K]"""

Yp = 0.24
"""Helium fraction [under development] [1]"""

Nnu = 3.
"""Number of effective massless neutrino species [under development] [1]"""

F = 1.14
"""??? [under development]"""

fDM = 0.0
"""??? [under development]"""


# Parameters used for numerics
#--------------------

pk_type = 'EH'
"""sets is the linear perturbations should be calculated using boltzman solver ('boltz') or approximations ('EH' for Einstein and Hu or 'BBKS') """

pk_nonlin_type = 'halofit'
"""sets if the nonlinear matter power spectrum should be calculated using the halofit fitting function ('halofit') or the revised halofit fitting function ('rev_halofit') """


#aini = 1.0e-7
aini=2.14956993e-07   # value to match COSMICS
"""a used as initial starting point for Boltzman calculation - Warning: aini may now be a function k for some settings [under development]"""

#intamin = -1.0
#"""mininum a value used for the interpolation routines (-1 sets to aini/1.1) [under development]"""

#intamax = 1.0
#"""maximum a value used for the interpolation routines [under development]"""

#intnum = 100
#""" number of points ussed in interpolation [under development]"""

#speed = "slow"
#"""fast (interpolations used) or slow - full calcs [under development]"""

recomb = "cosmics"
"""code to compute recombination: 'recfast++' or 'cosmics' [under development]"""

#TODO: fix this! use resources to resolve paths
cosmics_dir = "../Tests/comparison_files/cosmics/zend_0/"
"""COSMICS directory for recombination [under development]"""

omega_suppress = False
""" suppress radiation contribution in omega total as is often done """

cosmo_nudge=[1.,1.,1.]
""" nudge factors for H0, omega_gam, and omega_neu to compare with other codes - set to [1.,1.,1.] or leave out to suppress nudge"""