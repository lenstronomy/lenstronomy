__author__ = "aymgal"

import os
import json
import shutil
import numpy as np

from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler
import lenstronomy.Util.sampling_util as utils

__all__ = ["MultiNestSampler"]


class MultiNestSampler(NestedSampler):
    """Wrapper for nested sampling algorithm MultInest by F.

    Feroz & M. Hobson
    papers : arXiv:0704.3704, arXiv:0809.3437, arXiv:1306.2144
    pymultinest doc : https://johannesbuchner.github.io/PyMultiNest/pymultinest.html
    """

    def __init__(
        self,
        likelihood_module,
        prior_type="uniform",
        prior_means=None,
        prior_sigmas=None,
        width_scale=1,
        sigma_scale=1,
        output_dir=None,
        output_basename="-",
        remove_output_dir=False,
        use_mpi=False,
    ):
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
        :param use_mpi: flag directly passed to MultInest sampler (NOT TESTED)
        """
        self._check_install()
        super(MultiNestSampler, self).__init__(
            likelihood_module,
            prior_type,
            prior_means,
            prior_sigmas,
            width_scale,
            sigma_scale,
        )

        # here we assume number of dimensons = number of parameters
        self.n_params = self.n_dims

        if output_dir is None:
            self._output_dir = "multinest_out_default"
        else:
            self._output_dir = output_dir

        self._is_master = True
        self._use_mpi = use_mpi

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

        self.files_basename = os.path.join(self._output_dir, output_basename)

        # required for analysis : save parameter names in json file
        if self._is_master:
            with open(self.files_basename + "params.json", "w") as file:
                json.dump(self.param_names, file, indent=2)

        self._rm_output = remove_output_dir

    def run(self, kwargs_run):
        """Run the MultiNest nested sampler.

        see https://johannesbuchner.github.io/PyMultiNest/pymultinest.html for content
        of kwargs_run

        :param kwargs_run: kwargs directly passed to pymultinest.run
        :return: samples, means, logZ, logZ_err, logL, stats
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)

        if self._pymultinest_installed:
            self._pymultinest.run(
                self.log_likelihood,
                self.prior,
                self.n_dims,
                outputfiles_basename=self.files_basename,
                resume=False,
                verbose=True,
                init_MPI=self._use_mpi,
                **kwargs_run
            )

            analyzer = self._Analyzer(
                self.n_dims, outputfiles_basename=self.files_basename
            )
            samples = analyzer.get_equal_weighted_posterior()[:, :-1]
            data = analyzer.get_data()  # gets data from the *.txt output file
            stats = analyzer.get_stats()

        else:
            # in case MultiNest was not compiled properly, for unit tests
            samples = np.zeros((1, self.n_dims))
            data = np.zeros((self.n_dims, 3))
            stats = {
                "global evidence": np.zeros(self.n_dims),
                "global evidence error": np.zeros(self.n_dims),
                "modes": [{"mean": np.zeros(self.n_dims)}],
            }

        logL = -0.5 * data[:, 1]  # since the second data column is -2*logL
        logZ = stats["global evidence"]
        logZ_err = stats["global evidence error"]
        # or better to use stats['marginals'][:]['median'] ???
        means = stats["modes"][0]["mean"]

        print(
            "MultiNest output files have been saved to {}*".format(self.files_basename)
        )

        if self._rm_output and self._is_master:
            shutil.rmtree(self._output_dir, ignore_errors=True)
            print("MultiNest output directory removed")

        return samples, means, logZ, logZ_err, logL, stats

    def _check_install(self):
        try:
            import pymultinest
            from pymultinest.analyse import Analyzer
        except:
            print(
                "Warning : MultiNest/pymultinest not properly installed (results might be unexpected). \
                    You can get it from : https://johannesbuchner.github.io/PyMultiNest/pymultinest.html"
            )
            self._pymultinest_installed = False
        else:
            self._pymultinest_installed = True
            self._pymultinest = pymultinest
            self._Analyzer = Analyzer
