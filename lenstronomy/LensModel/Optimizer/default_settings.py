"""
This file contains default settings for the optimization class. If the settings are not explicitly specified in the
call to Optimizer, the setting will default to the parameters in this class
"""

# the tolerance on lens model source position (used to fit image positions)
default_tol_source = 1e-5
# the tolerance on lens model magnification (used to fit magnifications)
default_tol_mag = 0.2
# the tolerance on lens model centroid position (used to fit the macromodel centroid; usually can constrain the mass
# centroid to ~ 50 m.a.s.
default_tol_centroid = 0.05
# the tolerance for the Nelder-Mead downhill simplex optimization
default_tol_simplex = 1e-9

# The standard deviation of the particle swarm particles; when the last few particles have standard deviation below this,
# the algorithm terminates
default_pso_convergence_standardDEV = 0.01

# An alternative measure of swarm convergence: if the mean of the penalities associated with the last few lens models
# is below this value, the swarm terminates. Usually this terminates the particle swarm before the
# 'pso_convergence_standardDEV' criterion is met
default_pso_convergence_mean = 2

# The source position penalty below which the magnifications of the lens model will be calculated. This is useful for
# avoiding the expensive computation of image magnifications for lens models that don't fit the image positions.
default_pso_compute_magnification = 5

# setting for the particle swarm optimization
default_n_iterations = 250
default_n_particles = 50
