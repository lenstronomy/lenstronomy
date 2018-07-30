__author__ = 'dgilman'

import numpy as np
from PSO_optimizer import ParticleSwarmOptimizer
from params import Params
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from single_plane import SinglePlaneOptimizer
from multi_plane import MultiPlaneOptimizer
from scipy.optimize import minimize

class Optimizer(object):

    """
    class which executes the optimization routines. Currently implemented as a particle swarm optimization followed by
    a downhill simplex routine.

    Particle swarm optimizer is modified from the CosmoHammer particle swarm routine with a different convergence criterion.

    """

    def __init__(self, x_pos, y_pos, magnification_target=None, redshift_list=[], lens_model_list=[], kwargs_lens=[], optimizer_routine='optimize_SIE_shear',
                 multiplane=None, z_main = None, z_source=None,
                  tol_source=1e-5, tol_mag=0.3, tol_centroid=0.5, centroid_0=[0,0],
                 astropy_instance=None,
                 interpolate=False, verbose=False):
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
        """

        self.multiplane = multiplane
        self.verbose = verbose

        # make sure the length of observed positions matches, length of observed magnifications, etc.
        self._init_test(x_pos, y_pos, magnification_target, tol_source, redshift_list, lens_model_list, kwargs_lens,
                        z_source, z_main, multiplane, astropy_instance)

        # initialize lens model class
        lensModel = LensModelExtensions(lens_model_list=lens_model_list, redshift_list=redshift_list, z_source=z_source,
                                        cosmo=astropy_instance, multi_plane=multiplane)

        # initiate a params class that, based on the optimization routine, determines which parameters/lens models to optimize
        self.Params = Params(zlist=lensModel.redshift_list, lens_list=lensModel.lens_model_list, arg_list=kwargs_lens,
                             optimizer_routine=optimizer_routine)

        # initialize particle swarm inital param limits
        self.lower_limit = self.Params.tovary_lower_limit
        self.upper_limit = self.Params.tovary_upper_limit

        # initiate optimizer classes, one for particle swarm and one for the downhill simplex
        if multiplane is False:

            self.optimizer = SinglePlaneOptimizer(lensModel, x_pos, y_pos, tol_source, self.Params, \
                                                  magnification_target, tol_mag, centroid_0, tol_centroid,
                                                  k_start=self.Params.k_start, arg_list=kwargs_lens,verbose=verbose)

            self.optimizer_amoeba = SinglePlaneOptimizer(lensModel, x_pos, y_pos, tol_source, self.Params, \
                                                         magnification_target, tol_mag, centroid_0, tol_centroid,
                                                         k_start=self.Params.k_start, arg_list=kwargs_lens, mag_penalty=True,
                                                         return_mode='amoeba', verbose=verbose)


        else:

            self.optimizer = MultiPlaneOptimizer(lensModel, kwargs_lens, x_pos, y_pos, tol_source, self.Params,
                                                 magnification_target,
                                                 tol_mag, centroid_0, tol_centroid, z_main, z_source,
                                                 astropy_instance, interpolated=interpolate, verbose=verbose)

            self.optimizer_amoeba = MultiPlaneOptimizer(lensModel, kwargs_lens, x_pos, y_pos, tol_source, self.Params,
                                                        magnification_target,
                                                        tol_mag, centroid_0, tol_centroid, z_main, z_source,
                                                        astropy_instance, interpolated=interpolate, return_mode='amoeba', mag_penalty=True,
                                                        return_array=False, verbose=verbose)

    def optimize(self,n_particles=None,n_iterations=None):

        """

        :param n_particles: number of particle swarm particles
        :param n_iterations: number of particle swarm iternations
        total number of lens models sovled: n_particles*n_iterations
        :return: lens model keywords, [optimized source position], best fit image positions
        """

        # particle swarm optimization
        self.optimizer._init_particles(n_particles,n_iterations)
        optimized_PSO = self._pso(n_particles,n_iterations,self.optimizer)

        if self.verbose:
            print 'starting amoeba... '

        # downhill simplex optimization
        self.optimizer_amoeba._init_particles(n_particles, n_iterations)
        optimized_downhill_simplex = minimize(self.optimizer_amoeba,x0=optimized_PSO,method='Nelder-Mead',tol=1e-11)

        # combine the optimized parameters with the parameters kept fixed during the optimization to obtain full kwargs_lens
        kwargs_lens = self.Params.argstovary_todictionary(optimized_downhill_simplex['x']) + self.Params.argsfixed_todictionary()

        # solve for the optimized image positions
        ximg,yimg = self.optimizer_amoeba._get_images(kwargs_lens)

        return kwargs_lens, [self.optimizer_amoeba.source_x, self.optimizer_amoeba.source_y], [ximg, yimg]

    def _pso(self, n_particles, n_iterations, optimizer, lowerLimit=None, upperLimit=None, threadCount=1,social_influence = 0.9,
             personal_influence=1.3):

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

        if lowerLimit is None or upperLimit is None:
            lowerLimit, upperLimit = self.lower_limit, self.upper_limit

        else:
            lowerLimit = np.maximum(lowerLimit, self.lower_limit)
            upperLimit = np.minimum(upperLimit, self.upper_limit)

        pso = ParticleSwarmOptimizer(optimizer, lowerLimit, upperLimit, n_particles, threads=threadCount)

        #swarms, gBests = pso.optimize(maxIter=n_iterations,c1=social_influence,c2=personal_influence)
        swarms, gBests = pso.optimize(maxIter=n_iterations)

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
        assert len(zlist) == len(lens_list) == len(arg_list)

        if multiplane is True:
            assert z_source is not None
            assert z_main is not None
            assert astropy_instance is not None



