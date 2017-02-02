__author__ = 'sibirrer'

# Copyright (C) 2014 ETH Zurich, Institute for Astronomy


h = 0.6777
# h = 0.8
"""Hubble constant H0 = h*100km/s/Mpc """

omega_b = 0.0483
"""Baryon density (z=0)"""

omega_m = 0.2589+0.0483+0.0014
"""Matter density (dark + baryonic) (z=0)"""

omega_l_in = "flat"
"""Dark energy density. If flat then omega_l is 1.- omega_m - omega_r  """

w0 = -1.0
"""DE equation of state"""

wa = 0.0
"""DE equation of state evol: w(a)=w0+wa(1-a)"""

n = 0.9611
"""Spectral index"""

tau = 0.0952
"""Optical depth"""

pk_norm_type='sigma8'
"""Power spectrum normalisation scheme: 'deltah' for CMB normalisation or 'sigma8' for sigma8 normalisation"""

pk_norm = 0.8288
"""Powerspectrum Normalisation (z=0)"""

#deltah = 1843785.9626
# deltah = 4.6e-5
"""Powerspectrum Normalisation (early time)"""

Yp = 0.24
"""Helium fraction **need to incorporate**"""

Tcmb = 2.7255
"""CMB temperature [in kelvin] **need to incorporate**"""

Nnu = 3.046
"""Number of effective neutrino species **need to incorporate**"""

F = 1.14
"""***???*** **need to incorporate**"""

fDM = 0.0
"""***???*** **need to incorporate**"""

aini = 1.0e-7
"""a used as initial starting point for Boltzman calculation"""

intamin = -1
"""mininum a value used for the interpolation routines (-1 sets to aini/1.1)"""

intamax = 1.0
"""maximum a value used for the interpolation routines"""

intnum = 100
""" number of points ussed in interpolation"""

speed = "slow"
"""fast (interpolations used) or slow - full calcs """

recomb = "---"
"""code to compute recombination: 'recfast++' or 'cosmics'"""

cosmics_dir = "---"
"""COSMICS directory for recombination"""

pk_type = 'EH'
"""sets is the linear perturbations should be calculated using boltzman solver ('boltz') or approximations ('approx') """

pk_nonlin_type = 'rev_halofit'
"""sets if the nonlinear matter power spectrum should be calculated using the halofit fitting function ('halofit') or the revised halofit fitting function ('rev_halofit') """

omega_supress = False

cosmo_nudge = [1., 1., 1.]

tabulation = False