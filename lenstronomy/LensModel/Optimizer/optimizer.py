__author__ = 'dgilman'

import numpy as np
from lenstronomy.LensModel.Optimizer.particle_swarm import ParticleSwarmOptimizer
from lenstronomy.LensModel.Optimizer.params import Params
from lenstronomy.LensModel.Optimizer.single_background import SingleBackground
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Optimizer.single_plane import SinglePlaneLensing
from lenstronomy.LensModel.Optimizer.multi_plane import MultiPlaneLensing
from lenstronomy.LensModel.Optimizer.penalties import Penalties
from scipy.optimize import minimize
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from copy import deepcopy

class Optimizer(object):

    """
    class which executes the optimization routines. Currently implemented as a particle swarm optimization followed by
    a downhill simplex routine.

    Particle swarm optimizer is modified from the CosmoHammer particle swarm routine with different convergence criteria implemented.
    """

    def __init__(self, x_pos, y_pos, redshift_list=[], lens_model_list=[], kwargs_lens=[],
                 optimizer_routine='fixed_powerlaw_shear',magnification_target=None, multiplane=None,
                 z_main = None, z_source=None,tol_source=1e-5, tol_mag=0.2, tol_centroid=0.05, centroid_0=[0,0],
                 astropy_instance=None, verbose=False, re_optimize=False, particle_swarm=True,
                 pso_convergence_standardDEV=0.01, pso_convergence_mean=5, pso_compute_magnification=10,
                 tol_simplex_params=1e-3,tol_simplex_func = 1e-3,tol_src_penalty=0.5,constrain_params=None,
                 simplex_n_iterations=250, single_background=False, init_lensmodel = None, init_kwargs = None):


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
        :param pso_convergence_standardDEV: convergence criterion for particle swarm algorithm
        :param pso_convergence_mean: alternate convergence criterion for PSO; usually dominates the former
        :param pso_compute_magnification: flag for computing magnifications in the PSO; useful to avoid computing
        the magnification for lens models that are obviously wrong
        :param tol_simplex_func/params: tolerance for the scipy downhill simplex routine
        :param tol_src_penalty: if the source penalty is less than this value, the recomputation of the image positions
        will be skipped. (if the source penalty is good enough, you're guaranteed to match the input image positions so
        not point in recomputing them)
        :param constrain_params: additional parameters to constrain (type: dictionary)

        Format: {'parameter_name_1':[desired_value,uncertainty], 'parameter_name_2':[desired_value,uncertainty]}
        e.g.
        {'theta_E:[1,0.01]} will constrain the Einstein radius to 1 plus/minus 0.01
        The parameter name must be part of the 'params_to_vary' attribute of the specific optimization routine used
        (see class 'fixed_routines')

        Special cases are constraining shear and shear_pa in polar coordinates:
        {'shear':[0.05,0.01], 'shear_pa':[30,5]} will constrain the shear parameters in polar coordinates
        based on the cartesian e1/e2 values

        :param simplex_n_iterations: simplex_n_iterations times problem dimension gives the maximum # of iterations
        for the downhill simplex routine
        :param single_background: uses an approximation in which the path through background halos is only computed
        once; useful for models with a lot of background subhalos that are otherwise very computationally expensive to
        handle.

        Note: if running with particle_swarm = False, the re_optimize variable does nothing
        """

        self._multiplane = multiplane
        self._verbose = verbose
        x_pos, y_pos = np.array(x_pos), np.array(y_pos)
        self.x_pos,self.y_pos = x_pos,y_pos

        self._re_optimize = re_optimize
        self._particle_swarm = particle_swarm
        self._init_kwargs = kwargs_lens

        self._pso_convergence_standardDEV = pso_convergence_standardDEV
        self._tol_simplex_params = tol_simplex_params
        self._tol_simplex_func = tol_simplex_func
        self._tol_src_penalty = tol_src_penalty
        self._simplex_iter = simplex_n_iterations
        self._single_background = single_background

        # make sure the length of observed positions matches, length of observed magnifications, etc.
        self._init_test(x_pos, y_pos, magnification_target, tol_source, redshift_list, lens_model_list, kwargs_lens,
                        z_source, z_main, multiplane, astropy_instance)

        # initialize lens model class
        self._lensModel = LensModel(lens_model_list=lens_model_list, redshift_list=redshift_list,
                                    z_source=z_source,
                                    cosmo=astropy_instance, multi_plane=multiplane)

        # initiate a params class that, based on the optimization routine, determines which parameters/lens models to optimize
        self._params = Params(zlist=self._lensModel.redshift_list, lens_list=self._lensModel.lens_model_list, arg_list=kwargs_lens,
                              optimizer_routine=optimizer_routine, xpos=x_pos, ypos = y_pos)
        
        # initialize particle swarm inital param limits
        self._lower_limit, self._upper_limit = self._params.to_vary_limits(self._re_optimize)

        # initiate optimizer classes, one for particle swarm and one for the downhill simplex
        if multiplane is False:
            lensing_class = SinglePlaneLensing(self._lensModel, x_pos, y_pos, self._params, kwargs_lens)
            # don't bother with anything special here, just use the regular lensmodel class
            self.solver = LensEquationSolver(self._lensModel)

        else:
            if self._single_background:

                lensing_class = SingleBackground(self._lensModel, x_pos, y_pos, kwargs_lens, z_source, z_main,
                                                 astropy_instance, self._params.tovary_indicies, guess_lensmodel=
                                                 init_lensmodel, guess_kwargs=init_kwargs)
            else:
                lensing_class = MultiPlaneLensing(self._lensModel, x_pos, y_pos, kwargs_lens, z_source, z_main,
                                                    astropy_instance, self._params.tovary_indicies)

            self.solver = LensEquationSolver(lensing_class)

        self.lensModel = self.solver.lensModel

        self._optimizer = Penalties(tol_source, tol_mag, tol_centroid, lensing_class, centroid_0, magnification_target,
                                    params_to_constrain=constrain_params, param_class=self._params,
                                    pso_convergence_mean=pso_convergence_mean,
                                    pso_compute_magnification=pso_compute_magnification, compute_mags=False,
                                    verbose=verbose)

    def optimize(self, n_particles=50, n_iterations=250, restart=1):

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
        penalties, parameters, src_pen_best = [],[], []

        for run in range(0, restart):

            penalty, params = self._single_optimization(n_particles, n_iterations)
            penalties.append(penalty)
            parameters.append(params)
            src_pen_best.append(self._optimizer.src_pen_best)

        # select the best optimization
        best_index = np.argmin(penalties)

        # combine the optimized parameters with the parameters kept fixed during the optimization to obtain full kwargs_lens
        kwargs_varied = self._params.argstovary_todictionary(parameters[best_index])
        kwargs_lens_final = kwargs_varied + self._params.argsfixed_todictionary()

        # solve for the optimized image positions
        srcx, srcy = self._optimizer.lensing._ray_shooting_fast(kwargs_varied)
        source_x, source_y = np.mean(srcx), np.mean(srcy)

        # if we have a good enough solution, no point in recomputing the image positions since this can be quite slow
        # and will give the same answer
        if src_pen_best[best_index] < self._tol_src_penalty:
            x_image, y_image = self.x_pos, self.y_pos
        else:
            # Here, the solver has the instance of "lensing_class" or "LensModel" for multiplane/singleplane respectively.
            print('Warning: possibly a bad fit.')
            x_image, y_image = self.solver.image_position_from_source(source_x, source_y, kwargs_lens_final, arrival_time_sort = False)
        if self._verbose:
            print('optimization done.')
            print('Recovered source position: ', (srcx, srcy))

        return kwargs_lens_final, [source_x, source_y], [x_image, y_image]

    def _single_optimization(self, n_particles, n_iterations):

        self._optimizer._reset()
        self._optimizer._init_particles(n_particles, n_iterations)

        if self._particle_swarm:
            params = self._pso(n_particles, n_iterations, self._optimizer)

        else:
            params = self._params._kwargs_to_tovary(self._init_kwargs)

        if self._verbose:
            print('PSO done.')
            print('starting amoeba... ')

        # downhill simplex optimization
        self._optimizer._reset(compute_mags=True)
        options = {'adaptive': True, 'fatol': self._tol_simplex_func, 'xatol': self._tol_simplex_params,
                             'maxiter': self._simplex_iter * len(params)}

        optimized_downhill_simplex = minimize(self._optimizer, x0=params, method='Nelder-Mead',
                             options=options)

        penalty = self._optimizer._get_best()

        self._optimizer._reset()

        return penalty, optimized_downhill_simplex['x']

    def _pso(self, n_particles, n_iterations, optimizer):

        """
        :param n_particles: number of PSO particles
        :param n_iterations: number of PSO iterations
        :param optimizer: instance of SinglePlaneOptimizer or MultiPlaneOptimizer
        :return: optimized kwargs_lens
        """

        pso = ParticleSwarmOptimizer(optimizer, low=self._lower_limit, high=self._upper_limit, particleCount=n_particles)

        gBests = pso._optimize(maxIter=n_iterations,standard_dev=self._pso_convergence_standardDEV)

        likelihoods = [particle.fitness for particle in gBests]
        ind = np.argmax(likelihoods)

        return gBests[ind].position

    def _init_test(self,x_pos,y_pos,magnification_target,tol_source,zlist,lens_list,arg_list,
                   z_source,z_main,multiplane,astropy_instance):

        """
        check inputs
        """

        assert len(x_pos) == len(y_pos)
        assert len(magnification_target) == len(x_pos)
        assert tol_source is not None
        assert len(lens_list) == len(arg_list)

        if multiplane is True:
            assert len(zlist) == len(lens_list)
            assert z_source is not None
            assert z_main is not None
            assert astropy_instance is not None



