__author__ = 'dgilman'

import numpy as np
from lenstronomy.LensModel.Optimizer.particle_swarm import ParticleSwarmOptimizer
from lenstronomy.LensModel.Optimizer.params import Params
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Optimizer.single_plane import SinglePlaneOptimizer
from lenstronomy.LensModel.Optimizer.multi_plane import MultiPlaneOptimizer
from scipy.optimize import minimize

class Optimizer(object):

    """
    class which executes the optimization routines. Currently implemented as a particle swarm optimization followed by
    a downhill simplex routine.

    Particle swarm optimizer is modified from the CosmoHammer particle swarm routine with different convergence criteria implemented.
    """

    def __init__(self, x_pos, y_pos, redshift_list=[], lens_model_list=[], kwargs_lens=[], optimizer_routine='optimize_SIE_shear',
                 magnification_target=None, multiplane=None, z_main = None, z_source=None,
                 tol_source=1e-5, tol_mag=0.2, tol_centroid=None, centroid_0=[0,0],
                 astropy_instance=None, interpolate=False, verbose=False, re_optimize=False, particle_swarm=True,
                 pso_convergence_standardDEV=0.01, pso_convergence_mean=1, pso_compute_magnification=5,
                 tol_simplex=1e-10):

        """

        :param x_pos: observed position in arcsec
        :param y_pos: observed position in arcsec
        :param magnification_target: observed magnifications, uncertainty for the magnifications
        :param redshift_list: list of lens model redshifts
        :param lens_model_list: list of lens models
        :param kwargs_lens: keywords for lens models
        :param optimizer_routine: a set optimization routine; currently only 'optimize_SIE_shear' is implemented
        :param multiplane: multi-plane flag
        :param z_main: if multi-plane, macromodel redshift
        :param z_source: if multi-plane, macromodel redshift
        :param tol_source: tolerance on the source position in the optimization
        :param tol_mag: tolerance on the magnification
        :param tol_centroid: tolerance on the centroid of the mass distributions
        :param centroid_0: centroid of the optimized lens model (list [x,y])
        :param astropy_instance: instance of astropy
        :param interpolate: if multi-plane; flag to interpolate the background lens models
        :param verbose: flag to print status updates during optimization
        :param re_optimize: flag to run in re-optimization mode; the particle swarm particles will be initialized clustered
        around the values in kwargs_lens
        :param particle_swarm: if False, the optimizer will skip the particle swarm optimization and go straight to
        downhill simplex

        Note: if running with particle_swarm = False, the re_optimize variable does nothing
        """

        self.multiplane = multiplane
        self.verbose = verbose
        x_pos, y_pos = np.array(x_pos), np.array(y_pos)
        self._re_optimize = re_optimize
        self._particle_swarm = particle_swarm
        self._init_kwargs = kwargs_lens

        self._pso_convergence_standardDEV = pso_convergence_standardDEV
        self._tol_simplex = tol_simplex

        # make sure the length of observed positions matches, length of observed magnifications, etc.
        self._init_test(x_pos, y_pos, magnification_target, tol_source, redshift_list, lens_model_list, kwargs_lens,
                        z_source, z_main, multiplane, astropy_instance)

        # initialize lens model class
        lensModel = LensModelExtensions(lens_model_list=lens_model_list, redshift_list=redshift_list, z_source=z_source,
                                        cosmo=astropy_instance, multi_plane=multiplane)

        # initiate a params class that, based on the optimization routine, determines which parameters/lens models to optimize

        self.params = Params(zlist=lensModel.redshift_list, lens_list=lensModel.lens_model_list, arg_list=kwargs_lens,
                             optimizer_routine=optimizer_routine)

        # initialize particle swarm inital param limits

        self.lower_limit,self.upper_limit = self.params.to_vary_limits(self._re_optimize)

        # initiate optimizer classes, one for particle swarm and one for the downhill simplex
        if multiplane is False:

            self.optimizer = SinglePlaneOptimizer(lensModel, x_pos, y_pos, tol_source, self.params,
                                                  magnification_target, tol_mag, centroid_0, tol_centroid,
                                                  k_start=self.params.k_start, arg_list=kwargs_lens, verbose=verbose,
                                                  pso_convergence_mean=pso_convergence_mean,
                                                  pso_compute_magnification=pso_compute_magnification)

            self.optimizer_amoeba = SinglePlaneOptimizer(lensModel, x_pos, y_pos, tol_source, self.params,
                                                         magnification_target, tol_mag, centroid_0, tol_centroid,
                                                         k_start=self.params.k_start, arg_list=kwargs_lens, mag_penalty=True,
                                                         return_mode='amoeba', verbose=verbose,pso_convergence_mean=pso_convergence_mean,
                                                        pso_compute_magnification=pso_compute_magnification)


        else:

            self.optimizer = MultiPlaneOptimizer(lensModel, kwargs_lens, x_pos, y_pos, tol_source, self.params,
                                                 magnification_target,
                                                 tol_mag, centroid_0, tol_centroid, z_main, z_source,
                                                 astropy_instance, interpolated=interpolate, verbose=verbose,
                                                 pso_convergence_mean=pso_convergence_mean,
                                                 pso_compute_magnification=pso_compute_magnification)

            self.optimizer_amoeba = MultiPlaneOptimizer(lensModel, kwargs_lens, x_pos, y_pos, tol_source, self.params,
                                                        magnification_target,
                                                        tol_mag, centroid_0, tol_centroid, z_main, z_source,
                                                        astropy_instance, interpolated=interpolate, return_mode='amoeba',
                                                        mag_penalty=True,return_array=False, verbose=verbose,
                                                        pso_convergence_mean=pso_convergence_mean,
                                                        pso_compute_magnification=pso_compute_magnification)

    def optimize(self, n_particles=None, n_iterations=None, restart=1):

        """

        :param n_particles: number of particle swarm particles
        :param n_iterations: number of particle swarm iternations
        :param restart: number of times to execute the optimization;
        the best result of all optimizations will be returned.
        total number of lens models sovled: n_particles*n_iterations
        :return: lens model keywords, [optimized source position], best fit image positions
        """

        if restart < 0:
            raise ValueError("parameter 'restart' must be integer of value > 0")

        # particle swarm optimization
        penalties, parameters = [],[]

        for run in range(0, restart):

            penalty, params = self._single_optimization(n_particles, n_iterations)
            penalties.append(penalty)
            parameters.append(params)

        # select the best optimization
        best_index = np.argmin(penalties)

        # combine the optimized parameters with the parameters kept fixed during the optimization to obtain full kwargs_lens
        kwargs_varied = self.params.argstovary_todictionary(parameters[best_index])
        kwargs_lens_final = kwargs_varied + self.params.argsfixed_todictionary()

        # solve for the optimized image positions
        ximg,yimg,source_x,source_y = self.optimizer_amoeba._get_images(kwargs_varied)

        return kwargs_lens_final, [source_x, source_y], [ximg, yimg]

    def _single_optimization(self, n_particles, n_iterations):

        self.optimizer._init_particles(n_particles, n_iterations)

        if self._particle_swarm:
            params = self._single_PSO_optimization(n_particles, n_iterations)
        else:
            params = self.params._kwargs_to_tovary(self._init_kwargs)

        if self.verbose:
            print('starting amoeba... ')

        if self.multiplane:
            models, args = self.optimizer.multiplane_optimizer._get_interpolated_models()
            self.optimizer_amoeba.multiplane_optimizer.set_interpolated(models, args)
            self.optimizer_amoeba.multiplane_optimizer.inherit_rays(self.optimizer)

        # downhill simplex optimization
        self.optimizer_amoeba._init_particles(n_particles, n_iterations)
        optimized_downhill_simplex = minimize(self.optimizer_amoeba, x0=params, method='Nelder-Mead',
                                              tol=self._tol_simplex)

        penalty = self.optimizer_amoeba.get_best()
        parameters = optimized_downhill_simplex['x']

        self.optimizer.reset()
        self.optimizer_amoeba.reset()

        return penalty, parameters

    def _single_PSO_optimization(self, n_particles, n_iterations):

        optimized_PSO = self._pso(n_particles, n_iterations, self.optimizer)

        return optimized_PSO

    def _pso(self, n_particles, n_iterations, optimizer):

        """

        :param n_particles: number of PSO particles
        :param n_iterations: number of PSO iterations
        :param optimizer: instance of SinglePlaneOptimizer or MultiPlaneOptimizer
        :param lowerLimit: lower limit for PSO particles
        :param upperLimit: upper limit for PSO particles
        :param threadCount:
        :param social_influence:
        :param personal_influence:

        :return: optimized kwargs_lens
        """

        pso = ParticleSwarmOptimizer(optimizer, low=self.lower_limit, high=self.upper_limit, particleCount=n_particles)

        gBests = pso._optimize(maxIter=n_iterations,standard_dev=self._pso_convergence_standardDEV)

        likelihoods = [particle.fitness for particle in gBests]
        ind = np.argmax(likelihoods)

        return gBests[ind].position

    def _init_test(self,x_pos,y_pos,magnification_target,tol_source,zlist,lens_list,arg_list,
                   z_source,z_main,multiplane,astropy_instance):

        """
        check inputs
        """

        assert len(x_pos) == 4
        assert len(y_pos) == 4
        assert len(magnification_target) == len(x_pos)
        assert tol_source is not None
        assert len(lens_list) == len(arg_list)

        if multiplane is True:
            assert len(zlist) == len(lens_list)
            assert z_source is not None
            assert z_main is not None
            assert astropy_instance is not None



