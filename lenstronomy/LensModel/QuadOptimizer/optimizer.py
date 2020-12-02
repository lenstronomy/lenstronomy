__author__ = 'dgilman'

from scipy.optimize import minimize
import numpy as np
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from lenstronomy.LensModel.QuadOptimizer.multi_plane_fast import MultiplaneFast
from lenstronomy.Sampling.Pool.pool import choose_pool

__all__ = ['Optimizer']

class Optimizer(object):

    """
    class which executes the optimization routines. Currently implemented as a particle swarm optimization followed by
    a downhill simplex routine.

    Particle swarm optimizer is modified from the CosmoHammer particle swarm routine with different convergence criteria implemented.
    """

    def __init__(self, x_image, y_image, lens_model_list, redshift_list, z_lens, z_source,
                 parameter_class, astropy_instance=None, numerical_alpha_class=None,
                 particle_swarm=True, re_optimize=False, re_optimize_scale=1.,
                 pso_convergence_mean=50000, foreground_rays=None,
                 tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400):

        """

        :param x_image: x_image to fit (should be length 4)
        :param y_image: y_image to fit (should be length 4)
        :param lens_model_list: list of lens models for the system
        :param redshift_list: list of lens redshifts for the system
        :param z_lens: the main deflector redshift, the lens models being optimizer must be at this redshift
        :param z_source: the source redshift
        :param parameter_class: an instance of ParamClass (see documentation in QuadOptimizer.param_manager)
        :param astropy_instance: an instance of astropy to pass to the lens model
        :param numerical_alpha_class: a class to compute numerical deflection angles to pass to the lens model
        :param particle_swarm: bool, whether or not to use a PSO fit first
        :param re_optimize: bool, if True the initial spread of particles will be very tight
        :param re_optimize_scale: float, controls how tight the initial spread of particles is
        :param pso_convergence_mean: when to terminate the PSO fit
        :param foreground_rays: (optional) can pass in pre-computed foreground light rays from a previous fit
        so as to not waste time recomputing them
        :param tol_source: sigma in the source plane chi^2
        :param tol_simplex_func: tolerance for the downhill simplex optimization
        :param simplex_n_iterations: number of iterations per dimension for the downhill simplex optimization
        """

        self.fast_rayshooting = MultiplaneFast(x_image, y_image, z_lens, z_source,
                                                 lens_model_list, redshift_list, astropy_instance, parameter_class,
                                                 foreground_rays, tol_source, numerical_alpha_class)

        self._tol_source = tol_source

        self._pso_convergence_mean = pso_convergence_mean

        self._param_class = parameter_class

        self._tol_simplex_func = tol_simplex_func

        self._simplex_n_iterations = simplex_n_iterations

        self._particle_swarm = particle_swarm

        self._re_optimize = re_optimize
        self._re_optimize_scale = re_optimize_scale

    def optimize(self, n_particles=50, n_iterations=250, verbose=False, threadCount=1):

        """

        :param n_particles: number of PSO particles, will be ignored if self._particle_swarm is False
        :param n_iterations: number of PSO iterations, will be ignored if self._particle_swarm is False
        :param verbose: whether to print stuff
        :param pool: instance of Pool with a map method for multiprocessing
        :return: keyword arguments that map (x_image, y_image) to the same source coordinate (source_x, source_y)
        """

        if self._particle_swarm:

            if threadCount > 1:
                pool = choose_pool(mpi=False, processes=threadCount)
            else:
                pool = None

            kwargs = self._fit_pso(n_particles, n_iterations, pool, verbose)

        else:
            kwargs = self._param_class.kwargs_lens

        kwargs_lens_final, source_penalty = self._fit_amoeba(kwargs, verbose)

        args_lens_final = self._param_class.kwargs_to_args(kwargs_lens_final)
        source_x_array, source_y_array = self.fast_rayshooting.ray_shooting_fast(args_lens_final)
        source_x, source_y = np.mean(source_x_array), np.mean(source_y_array)

        if verbose:
            print('optimization done.')
            print('Recovered source position: ', (source_x_array, source_y_array))

        return kwargs_lens_final, [source_x, source_y]

    def _fit_pso(self, n_particles, n_iterations, pool, verbose):

        """
        Executes the PSO
        """

        low_bounds, high_bounds = self._param_class.bounds(self._re_optimize, self._re_optimize_scale)

        pso = ParticleSwarmOptimizer(self.fast_rayshooting.logL, low_bounds, high_bounds, n_particles,
                                     pool, args=[self._tol_source])

        best, info = pso.optimize(n_iterations, verbose, early_stop_tolerance=self._pso_convergence_mean)

        if verbose:
            print('PSO done... ')
            print('source plane chi^2: ', self.fast_rayshooting.source_plane_chi_square(best))
            print('total chi^2: ', self.fast_rayshooting.chi_square(best))

        kwargs = self._param_class.args_to_kwargs(best)

        return kwargs

    def _fit_amoeba(self, kwargs, verbose):

        """
        Executes the downhill simplex
        """

        args_init = self._param_class.kwargs_to_args(kwargs)

        options = {'adaptive': True, 'fatol': self._tol_simplex_func,
                   'maxiter': self._simplex_n_iterations * len(args_init)}

        method = 'Nelder-Mead'

        if verbose:
            print('starting amoeba... ')

        opt = minimize(self.fast_rayshooting.chi_square, x0=args_init,
                       method=method, options=options)

        kwargs = self._param_class.args_to_kwargs(opt['x'])
        source_penalty = opt['fun']

        return kwargs, source_penalty






