__author__ = 'aymgal'

import os
import shutil
import numpy as np

import lenstronomy.Util.sampling_util as utils


class DyPolyChordSampler(object):
    """
    Wrapper for dynamical nested sampling algorithm DyPolyChord
    by E. Higson, M. Hobson, W. Handley, A. Lasenby

    papers : arXiv:1704.03459, arXiv:1804.06406
    doc : https://dypolychord.readthedocs.io
    """

    def __init__(self, likelihood_module, prior_type='uniform', 
                 prior_means=None, prior_sigmas=None,
                 output_dir=None, output_basename='-', seed_increment=1,
                 remove_output_dir=False): #, use_mpi=False, num_mpi_procs=1):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param output_dir: name of the folder that will contain output files
        :param output_basename: prefix for output files
        :param remove_output_dir: remove the output_dir folder after completion
        :param seed_increment: seed increment for random number generator
        """
        self._check_install()

        self._ll = likelihood_module
        self.lowers, self.uppers = self._ll.param_limits
        self.n_dims, self.param_names = self._ll.param.num_param()

        if prior_type == 'gaussian':
            if prior_means is None or prior_sigmas is None:
                raise ValueError("For gaussian prior type, means and sigmas are required")
            self.means, self.sigmas = prior_means, prior_sigmas
        elif prior_type != 'uniform':
            raise ValueError("Sampling type {} not supported".format(prior_type))
        self.prior_type = prior_type

        self._output_dir= output_dir
        if os.path.exists(self._output_dir):
            shutil.rmtree(self._output_dir, ignore_errors=True)
        os.mkdir(self._output_dir)

        self.settings = {
            'file_root': output_basename,
            'base_dir': self._output_dir,
            'seed': seed_increment,
        }

        # if use_mpi:
        #     mpi_str = 'mpirun -np {}'.format(num_mpi_procs)
        # else:
        #     mpi_str = None

        if self._all_installed:
            # create the dyPolyChord callable object
            self._sampler = self._RunPyPolyChord(self.log_likelihood, 
                                                 self.prior, self.n_dims)
        else:
            self._sampler = None

        self._rm_output = remove_output_dir


    def prior(self, cube):
        """
        compute the mapping between the unit cube and parameter cube

        'copy=True' below because cube can not be modified in-place (read-only)

        :param cube: unit hypercube, sampled by the algorithm
        :return: hypercube in parameter space
        """
        if self.prior_type == 'gaussian':
            p = utils.cube2args_gaussian(cube, self.lowers, self.uppers,
                                         self.means, self.sigmas, self.n_dims,
                                         copy=True)
        elif self.prior_type == 'uniform':
            p = utils.cube2args_uniform(cube, self.lowers, self.uppers, 
                                        self.n_dims, copy=True)
        return p


    def log_likelihood(self, args):
        """
        compute the log-likelihood given list of parameters

        :param args: parameter values
        :return: log-likelihood (from the likelihood module)
        """
        phi = []
        logL, _ = self._ll.likelihood(args)
        if not np.isfinite(logL):
            print("WARNING : logL is not finite : return very low value instead")
            logL = -1e15
        return float(logL), phi


    def run(self, dynamic_goal, kwargs_run):
        """
        run the DyPolyChord dynamical nested sampler

        see https://dypolychord.readthedocs.io for content of kwargs_run

        :param dynamic_goal: 0 for evidence computation, 1 for posterior computation
        :param kwargs_run: kwargs directly passed to dyPolyChord.run_dypolychord
        :return: samples, means, logZ, logZ_err, logL
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)
        
        if self._all_installed:
            # TODO : put a default dynamic_goal ?
            # dynamic_goal = 0 for evidence-only, 1 for posterior-only

            self._dyPolyChord.run_dypolychord(self._sampler, dynamic_goal, 
                                              self.settings, **kwargs_run)

            results = self._process_run(self.settings['file_root'], 
                                        self.settings['base_dir'])
            stats   = self._process_stats(self.settings['file_root'], 
                                          self.settings['base_dir'])

            samples = results['theta']
            logL    = results['logl']
            logZ     = stats['logZ']
            logZ_err = stats['logZerr']
            means    = stats['param_means']

            # ALTERNATIVE WAY :
            # logZ = self._estim.logz(results)
            # means = np.array([self._estim.param_mean(results, param_ind=i) for i in range(self.n_dims)])
            # TODO : check if it is equal to the other way above
            print('The log evidence estimate using the first run is {}'
                  .format(logZ))
            print('The estimated mean of the first parameter is {}'
                  .format(means[0]))
        
        else:
            # in case DyPolyChord or NestCheck was not compiled properly, for unit tests
            samples = np.zeros((1, self.n_dims))
            means = np.zeros(self.n_dims)
            logL = np.zeros(self.n_dims)
            logZ = np.zeros(self.n_dims)
            logZ_err = np.zeros(self.n_dims)

        if self._rm_output:
            shutil.rmtree(self._output_dir, ignore_errors=True)

        return samples, means, logZ, logZ_err, logL


    def _check_install(self):
        try:
            import dyPolyChord
            from dyPolyChord import pypolychord_utils
        except:
            print("Warning : dyPolyChord not properly installed. \
You can get it from : https://github.com/ejhigson/dyPolyChord")
            dypolychord_installed = False
        else:
            dypolychord_installed = True
            self._dyPolyChord = dyPolyChord
            self._RunPyPolyChord = pypolychord_utils.RunPyPolyChord

        try:
            from nestcheck import data_processing
            from nestcheck import estimators
        except:
            print("Warning : nestcheck not properly installed. \
You can get it from : https://github.com/ejhigson/nestcheck")
            nestcheck_installed = False
        else:
            nestcheck_installed = True
            self._process_run = data_processing.process_polychord_run
            self._process_stats = data_processing.process_polychord_stats
            self._estim = estimators

        self._all_installed = dypolychord_installed and nestcheck_installed

