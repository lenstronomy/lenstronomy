__author__ = 'aymgal'

import os
import shutil
import numpy as np

from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler
import lenstronomy.Util.sampling_util as utils


class DyPolyChordSampler(NestedSampler):
    """
    Wrapper for dynamical nested sampling algorithm DyPolyChord
    by E. Higson, M. Hobson, W. Handley, A. Lasenby

    papers : arXiv:1704.03459, arXiv:1804.06406
    doc : https://dypolychord.readthedocs.io
    """

    def __init__(self, likelihood_module, prior_type='uniform', 
                 prior_means=None, prior_sigmas=None, width_scale=1, sigma_scale=1,
                 output_dir=None, output_basename='-', seed_increment=1,
                 remove_output_dir=False, use_mpi=False): #, num_mpi_procs=1):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param width_scale: scale the widths of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by this factor
        :param output_dir: name of the folder that will contain output files
        :param output_basename: prefix for output files
        :param remove_output_dir: remove the output_dir folder after completion
        :param seed_increment: seed increment for random number generator
        :param use_mpi: Use MPI computing if `True`
        """
        self._check_install()
        super(DyPolyChordSampler, self).__init__(likelihood_module, prior_type, 
                                                 prior_means, prior_sigmas,
                                                 width_scale, sigma_scale)

        # if use_mpi:
        #     mpi_str = 'mpirun -np {}'.format(num_mpi_procs)
        # else:
        #     mpi_str = None

        self._use_mpi = use_mpi

        self._output_dir= output_dir
        self._is_master = True

        if self._use_mpi:
            from mpi4py import MPI
            self._comm = MPI.COMM_WORLD

            if self._comm.Get_rank() != 0:
                self._is_master = False
        else:
            self._comm = None

        if self._is_master:
            if os.path.exists(self._output_dir):
                shutil.rmtree(self._output_dir, ignore_errors=True)
            os.mkdir(self._output_dir)

        self._output_basename = output_basename
        self.settings = {
            'file_root': self._output_basename,
            'base_dir': self._output_dir,
            'seed': seed_increment,
        }

        if self._all_installed:
            # create the dyPolyChord callable object
            self._sampler = self._RunPyPolyChord(self.log_likelihood,
                                                 self.prior, self.n_dims)
        else:
            self._sampler = None

        self._rm_output = remove_output_dir
        self._has_warned = False

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
            if not self._has_warned:
                print("WARNING : logL is not finite : return very low value instead")
            logL = -1e15
            self._has_warned = True
        return float(logL), phi

    def run(self, dynamic_goal, kwargs_run):
        """
        run the DyPolyChord dynamical nested sampler

        see https://dypolychord.readthedocs.io for content of kwargs_run

        :param dynamic_goal: 0 for evidence computation, 1 for posterior computation
        :param kwargs_run: kwargs directly passed to dyPolyChord.run_dypolychord
        :return: samples, means, logZ, logZ_err, logL, ns_run
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)

        if self._all_installed:
            # TODO : put a default dynamic_goal ?
            # dynamic_goal = 0 for evidence-only, 1 for posterior-only

            self._dyPolyChord.run_dypolychord(self._sampler, dynamic_goal,
                                              self.settings,
                                              comm=self._comm, **kwargs_run)

            if self._is_master:
                ns_run = self._ns_process_run(self.settings['file_root'],
                                           self.settings['base_dir'])

        else:
            # in case DyPolyChord or NestCheck was not compiled properly, for unit tests
            ns_run = {
                'theta': np.zeros((1, self.n_dims)),
                'logl': np.zeros(self.n_dims),
                'output': {
                    'logZ': np.zeros(self.n_dims),
                    'logZerr': np.zeros(self.n_dims),
                    'param_means': np.zeros(self.n_dims)
                }
            }
            self._write_equal_weights(ns_run['theta'], ns_run['logl'])

        if self._is_master:
            samples, logL = self._get_equal_weight_samples()
            # logL     = ns_run['logl']
            # samples_w = ns_run['theta']
            logZ     = ns_run['output']['logZ']
            logZ_err = ns_run['output']['logZerr']
            means    = ns_run['output']['param_means']

            print('The log evidence estimate using the first run is {}'
                  .format(logZ))
            print('The estimated mean of the first parameter is {}'
                  .format(means[0]))

            if self._rm_output:
                shutil.rmtree(self._output_dir, ignore_errors=True)

            return samples, means, logZ, logZ_err, logL, ns_run
        else:
            return None

    def _get_equal_weight_samples(self):
        """
        Inspired by pymultinest's Analyzer,
        because DyPolyChord has more or less the same output conventions as MultiNest
        """
        file_name = '{}_equal_weights.txt'.format(self._output_basename)
        file_path = os.path.join(self._output_dir, file_name)
        data = np.loadtxt(file_path, ndmin=2)
        logL = -0.5 * data[:, 0]
        samples = data[:, 1:]
        return samples, logL

    def _write_equal_weights(self, samples, logL):
        # write fake output file for unit tests
        file_name = '{}_equal_weights.txt'.format(self._output_basename)
        file_path = os.path.join(self._output_dir, file_name)
        data = np.zeros((samples.shape[0], 1+samples.shape[1]))
        data[:, 0]  = -2. * logL
        data[:, 1:] = samples
        np.savetxt(file_path, data, fmt='% .14E')

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
        except:
            print("Warning : nestcheck not properly installed (results might be unexpected). \
You can get it from : https://github.com/ejhigson/nestcheck")
            nestcheck_installed = False
        else:
            nestcheck_installed = True
            self._ns_process_run = data_processing.process_polychord_run

        self._all_installed = dypolychord_installed and nestcheck_installed
