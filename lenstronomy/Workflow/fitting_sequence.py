from lenstronomy.Workflow.psf_fitting import PsfFitting
from lenstronomy.Workflow.alignment_matching import AlignmentFitting
from lenstronomy.Workflow.flux_calibration import FluxCalibration
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Workflow.multi_band_manager import MultiBandUpdateManager
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.sampler import Sampler
from lenstronomy.Sampling.Samplers.multinest_sampler import MultiNestSampler
from lenstronomy.Sampling.Samplers.polychord_sampler import DyPolyChordSampler
from lenstronomy.Sampling.Samplers.dynesty_sampler import DynestySampler
from lenstronomy.Sampling.Samplers.nautilus_sampler import NautilusSampler
from lenstronomy.Sampling.Samplers.cobaya_sampler import CobayaSampler
import numpy as np
import lenstronomy.Util.analysis_util as analysis_util

__all__ = ["FittingSequence"]


class FittingSequence(object):
    """Class to define a sequence of fitting applied, inherit the Fitting class this is
    a Workflow manager that allows to update model configurations before executing
    another step in the modelling The user can take this module as an example of how to
    create their own workflows or build their own around the FittingSequence."""

    def __init__(
        self,
        kwargs_data_joint,
        kwargs_model,
        kwargs_constraints,
        kwargs_likelihood,
        kwargs_params,
        mpi=False,
        verbose=True,
    ):
        """

        :param kwargs_data_joint: keyword argument specifying the data according to LikelihoodModule
        :param kwargs_model: keyword arguments to describe all model components used in
         class_creator.create_class_instances()
        :param kwargs_constraints: keyword arguments of the Param() class to handle parameter constraints during the
         sampling (except upper and lower limits and sampling input mean and width)
        :param kwargs_likelihood: keyword arguments of the Likelihood() class to handle parameters and settings of the
         likelihood
        :param kwargs_params: setting of the sampling bounds and initial guess mean and spread.
         The argument is organized as:
         'lens_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'lens_light_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'point_source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'extinction_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'special': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
         'tracer_source_model': [kwargs_init, kwargs_sigma, kwargs_fixed, kwargs_lower, kwargs_upper]
        :param mpi: MPI option (bool), if True, will launch an MPI Pool job for the steps in the fitting sequence where
         possible
        :param verbose: bool, if True prints temporary results and indicators of the fitting process
        """
        self.kwargs_data_joint = kwargs_data_joint
        self.multi_band_list = kwargs_data_joint.get("multi_band_list", [])
        self.multi_band_type = kwargs_data_joint.get("multi_band_type", "single-band")
        self._verbose = verbose
        self._mpi = mpi
        self._updateManager = MultiBandUpdateManager(
            kwargs_model,
            kwargs_constraints,
            kwargs_likelihood,
            kwargs_params,
            num_bands=len(self.multi_band_list),
        )
        self._mcmc_init_samples = None

    @property
    def kwargs_fixed(self):
        """Returns the updated kwargs_fixed from the update manager.

        :return: list of fixed kwargs, see UpdateManager()
        """
        return self._updateManager.fixed_kwargs

    def fit_sequence(self, fitting_list):
        """

        :param fitting_list: list of [['string', {kwargs}], ..] with 'string being the specific fitting option and
         kwargs being the arguments passed to this option
        :return: fitting results
        """
        chain_list = []
        for i, fitting in enumerate(fitting_list):
            fitting_type = fitting[0]
            kwargs = fitting[1]

            if fitting_type == "restart":
                self._updateManager.set_init_state()

            elif fitting_type == "update_settings":
                self.update_settings(**kwargs)

            elif fitting_type == "set_param_value":
                self.set_param_value(**kwargs)

            elif fitting_type == "fix_not_computed":
                self.fix_not_computed(**kwargs)

            elif fitting_type == "psf_iteration":
                self.psf_iteration(**kwargs)

            elif fitting_type == "align_images":
                self.align_images(**kwargs)

            elif fitting_type == "calibrate_images":
                self.flux_calibration(**kwargs)

            elif fitting_type == "PSO":
                kwargs_result, chain, param = self.pso(**kwargs)
                self._updateManager.update_param_state(**kwargs_result)

                chain_list.append([fitting_type, chain, param])

            elif fitting_type == "SIMPLEX":
                kwargs_result = self.simplex(**kwargs)
                self._updateManager.update_param_state(**kwargs_result)
                chain_list.append([fitting_type, kwargs_result])

            elif fitting_type in ["MCMC", "emcee", "zeus"]:
                if fitting_type == "MCMC":
                    print("MCMC selected. Sampling with default option emcee.")
                    fitting_type = "emcee"
                if "init_samples" not in kwargs:
                    kwargs["init_samples"] = self._mcmc_init_samples
                elif kwargs["init_samples"] is None:
                    kwargs["init_samples"] = self._mcmc_init_samples
                mcmc_output = self.mcmc(**kwargs, sampler_type=fitting_type)
                kwargs_result = self._result_from_mcmc(mcmc_output)
                self._updateManager.update_param_state(**kwargs_result)
                chain_list.append(mcmc_output)

            elif fitting_type == "Cobaya":
                print("Using the Metropolis--Hastings MCMC sampler in Cobaya.")
                param_class = self.param_class
                kwargs_temp = self._updateManager.parameter_state
                mean_start = param_class.kwargs2args(**kwargs_temp)
                kwargs_sigma = self._updateManager.sigma_kwargs
                sigma_start = np.array(param_class.kwargs2args(**kwargs_sigma))
                # pass the likelihood and starting info to the sampler
                sampler = CobayaSampler(self.likelihoodModule, mean_start, sigma_start)
                # run the sampler
                updated_info, sampler_type, best_fit_values = sampler.run(**kwargs)
                # change the best-fit values returned by cobaya into lenstronomy kwargs format
                best_fit_kwargs = self.param_class.args2kwargs(
                    best_fit_values, bijective=True
                )
                # collect the products
                mh_output = [updated_info, sampler_type, best_fit_kwargs]
                # append the products to the chain list
                chain_list.append(mh_output)

            elif fitting_type in [
                "dynesty",
                "dyPolyChord",
                "MultiNest",
                "nested_sampling",
            ]:
                if fitting_type == "nested_sampling":
                    print(
                        "Nested sampling selected. Sampling with default option dynesty."
                    )
                    fitting_type = "dynesty"
                ns_output = self.nested_sampling(**kwargs, sampler_type=fitting_type)
                chain_list.append(ns_output)

            elif fitting_type == "Nautilus":
                # do importance nested sampling with Nautilus
                nautilus = NautilusSampler(
                    likelihood_module=self.likelihoodModule, mpi=self._mpi, **kwargs
                )
                points, log_w, log_l, log_z = nautilus.run(**kwargs)
                chain_list.append([points, log_w, log_l, log_z])
                if kwargs.get("verbose", False):
                    print(len(points), "number of points sampled")
                kwargs_result = self.best_fit_from_samples(points, log_l)
                self._updateManager.update_param_state(**kwargs_result)

            else:
                raise ValueError(
                    "fitting_sequence {} is not supported. Please use: 'PSO', 'SIMPLEX', "
                    "'MCMC' or 'emcee', 'zeus', 'Cobaya', "
                    "'dynesty', 'dyPolyChord',  'Multinest', 'Nautilus, '"
                    "'psf_iteration', 'restart', 'update_settings', 'calibrate_images' or "
                    "'align_images'".format(fitting_type)
                )

        return chain_list

    def best_fit(self, bijective=False):
        """

        :param bijective: bool, if True, the mapping of image2source_plane and the mass_scaling parameterisation are inverted. If you do not use those options, there is no effect.
        :return: best fit model of the current state of the FittingSequence class
        """

        return self._updateManager.best_fit(bijective=bijective)

    def update_state(self, kwargs_update):
        """Updates current best fit state to the input model keywords specified.

        :param kwargs_update: format of kwargs_result
        :return: None
        """
        self._updateManager.update_param_state(**kwargs_update)

    @property
    def best_fit_likelihood(self):
        """Returns the log likelihood of the best fit model of the current state of this
        class.

        :return: log likelihood, float
        """
        kwargs_result = self.best_fit(bijective=False)
        param_class = self.param_class
        likelihoodModule = self.likelihoodModule
        logL = likelihoodModule.logL(param_class.kwargs2args(**kwargs_result))
        return logL

    @property
    def bic(self):
        """Bayesian information criterion (BIC) of the model.

        :return: bic value, float
        """
        num_data = self.likelihoodModule.num_data
        num_param_nonlinear = self.param_class.num_param()[0]
        num_param_linear = self.param_class.num_param_linear()
        num_param = num_param_nonlinear + num_param_linear
        bic = analysis_util.bic_model(self.best_fit_likelihood, num_data, num_param)
        return bic

    @property
    def param_class(self):
        """

        :return: Param() class instance reflecting the current state of FittingSequence
        """
        return self._updateManager.param_class

    @property
    def likelihoodModule(self):
        """

        :return: Likelihood() class instance reflecting the current state of FittingSequence
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihoodModule = LikelihoodModule(
            self.kwargs_data_joint, kwargs_model, self.param_class, **kwargs_likelihood
        )
        return likelihoodModule

    def simplex(self, n_iterations, method="Nelder-Mead"):
        """Downhill simplex optimization using the Nelder-Mead algorithm.

        :param n_iterations: maximum number of iterations to perform
        :param method: the optimization method used, see documentation in
            scipy.optimize.minimize
        :return: result of the best fit
        """

        param_class = self.param_class
        kwargs_temp = self._updateManager.parameter_state
        init_pos = param_class.kwargs2args(**kwargs_temp)
        sampler = Sampler(likelihoodModule=self.likelihoodModule)
        result = sampler.simplex(init_pos, n_iterations, method)

        kwargs_result = param_class.args2kwargs(result, bijective=True)
        return kwargs_result

    def mcmc(
        self,
        n_burn,
        n_run,
        walkerRatio=None,
        n_walkers=None,
        sigma_scale=1,
        threadCount=1,
        init_samples=None,
        re_use_samples=True,
        sampler_type="EMCEE",
        progress=True,
        backend_filename=None,
        start_from_backend=False,
        **kwargs_zeus
    ):
        """MCMC routine.

        :param n_burn: number of burn in iterations (will not be saved)
        :param n_run: number of MCMC iterations that are saved
        :param walkerRatio: ratio of walkers/number of free parameters
        :param n_walkers: integer, number of walkers of emcee (optional, if set, overwrites the walkerRatio input
        :param sigma_scale: scaling of the initial parameter spread relative to the width in the initial settings
        :param threadCount: number of CPU threads. If MPI option is set, threadCount=1
        :param init_samples: initial sample from where to start the MCMC process
        :param re_use_samples: bool, if True, re-uses the samples described in init_samples.nOtherwise starts from
         scratch.
        :param sampler_type: string, which MCMC sampler to be used. Options are 'emcee' and 'zeus'
        :param progress: boolean, if True shows progress bar in EMCEE
        :param backend_filename: name of the HDF5 file where sampling state is saved (through emcee backend engine)
        :type backend_filename: string
        :param start_from_backend: if True, start from the state saved in `backup_filename`.
         O therwise, create a new backup file with name `backup_filename` (any already existing file is overwritten!).
        :type start_from_backend: bool
        :param kwargs_zeus: zeus-specific kwargs
        :return: list of output arguments, e.g. MCMC samples, parameter names, logL distances of all samples specified
         by the specific sampler used
        """
        param_class = self.param_class
        # run PSO
        mcmc_class = Sampler(likelihoodModule=self.likelihoodModule)
        kwargs_temp = self._updateManager.parameter_state
        mean_start = param_class.kwargs2args(**kwargs_temp)
        kwargs_sigma = self._updateManager.sigma_kwargs
        sigma_start = np.array(param_class.kwargs2args(**kwargs_sigma)) * sigma_scale
        num_param, param_list = param_class.num_param()
        if n_walkers is None:
            if walkerRatio is None:
                raise ValueError(
                    "MCMC sampler needs either n_walkers or walkerRatio as input argument"
                )
            n_walkers = num_param * walkerRatio
        # run MCMC
        if init_samples is not None and re_use_samples is True:
            num_samples, num_param_prev = np.shape(init_samples)
            if num_param_prev == num_param:
                print("re-using previous samples to initialize the next MCMC run.")
                idxs = np.random.choice(len(init_samples), n_walkers)
                initpos = init_samples[idxs]
            else:
                raise ValueError(
                    "Can not re-use previous MCMC samples as number of parameters have changed!"
                )
        else:
            initpos = None

        if sampler_type == "zeus":
            # check if zeus is specified, if not default to emcee
            samples, dist = mcmc_class.mcmc_zeus(
                n_walkers,
                n_run,
                n_burn,
                mean_start,
                sigma_start,
                mpi=self._mpi,
                threadCount=threadCount,
                progress=progress,
                initpos=initpos,
                backend_filename=backend_filename,
                **kwargs_zeus
            )
            output = [sampler_type, samples, param_list, dist]
        else:
            # sample with emcee
            samples, dist = mcmc_class.mcmc_emcee(
                n_walkers,
                n_run,
                n_burn,
                mean_start,
                sigma_start,
                mpi=self._mpi,
                threadCount=threadCount,
                progress=progress,
                initpos=initpos,
                backend_filename=backend_filename,
                start_from_backend=start_from_backend,
            )
            output = [sampler_type, samples, param_list, dist]

        self._mcmc_init_samples = samples  # overwrites previous samples to continue from there in the next MCMC run
        return output

    def pso(
        self, n_particles, n_iterations, sigma_scale=1, print_key="PSO", threadCount=1
    ):
        """Particle Swarm Optimization.

        :param n_particles: number of particles in the Particle Swarm Optimization
        :param n_iterations: number of iterations in the optimization process
        :param sigma_scale: scaling of the initial parameter spread relative to the
            width in the initial settings
        :param print_key: string, printed text when executing this routine
        :param threadCount: number of CPU threads. If MPI option is set, threadCount=1
        :return: result of the best fit, the PSO chain of the best fit parameter after
            each iteration [lnlikelihood, parameters, velocities], list of parameters in
            same order as in chain
        """

        param_class = self.param_class
        kwargs_temp = self._updateManager.parameter_state
        init_pos = param_class.kwargs2args(**kwargs_temp)
        kwargs_sigma = self._updateManager.sigma_kwargs
        sigma_start = param_class.kwargs2args(**kwargs_sigma)
        lower_start = np.array(init_pos) - np.array(sigma_start) * sigma_scale
        upper_start = np.array(init_pos) + np.array(sigma_start) * sigma_scale

        num_param, param_list = param_class.num_param()
        # run PSO
        sampler = Sampler(likelihoodModule=self.likelihoodModule)
        result, chain = sampler.pso(
            n_particles,
            n_iterations,
            lower_start,
            upper_start,
            init_pos=init_pos,
            threadCount=threadCount,
            mpi=self._mpi,
            print_key=print_key,
            verbose=self._verbose,
        )
        kwargs_result = param_class.args2kwargs(result, bijective=True)
        return kwargs_result, chain, param_list

    def nested_sampling(
        self,
        sampler_type="dynesty",
        kwargs_run={},
        prior_type="uniform",
        width_scale=1,
        sigma_scale=1,
        output_basename="chain",
        remove_output_dir=True,
        dypolychord_dynamic_goal=0.8,
        polychord_settings={},
        dypolychord_seed_increment=200,
        output_dir="nested_sampling_chains",
        dynesty_bound="multi",
        dynesty_sample="auto",
    ):
        """Run (Dynamic) Nested Sampling algorithms, depending on the type of algorithm.

        :param sampler_type: 'MULTINEST', 'DYPOLYCHORD', 'DYNESTY'
        :param kwargs_run: keywords passed to the core sampling method
        :param prior_type: 'uniform' of
            'gaussian', for converting the unit hypercube to param cube :param width_scale:
            scale the width (lower/upper limits) of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by
            this factor
        :param output_basename: name of the folder in which the core
            MultiNest/PolyChord code will save output files
        :param remove_output_dir: if True, the above folder is removed after completion
        :param dypolychord_dynamic_goal: dynamic goal for DyPolyChord (trade-off between
            evidence (0) and posterior (1) computation) :param polychord_settings: settings
            dictionary to send to pypolychord. Check dypolychord documentation for details.
        :param dypolychord_seed_increment: seed increment for dypolychord with MPI.
            Check dypolychord documentation for details.
        :param dynesty_bound: see https://dynesty.readthedocs.io
        :param sampler_type: 'MULTINEST', 'DYPOLYCHORD', 'DYNESTY'
        :param kwargs_run: keywords passed to the core sampling method
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to
            param cube
        :param width_scale: scale the width (lower/upper limits) of the parameters space
            by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by
            this factor
        :param output_basename: name of the folder in which the core MultiNest/PolyChord
            code will save output files
        :param remove_output_dir: if True, the above folder is removed after completion
        :param dypolychord_dynamic_goal: dynamic goal for DyPolyChord (trade-off between
            evidence (0) and posterior (1) computation)
        :param polychord_settings: settings dictionary to send to pypolychord. Check
            dypolychord documentation for details.
        :param dypolychord_seed_increment: seed increment for dypolychord with MPI.
            Check dypolychord documentation for details.
        :param dynesty_bound: see https://dynesty.readthedocs.io for details
        :param dynesty_sample: see https://dynesty.readthedocs.io for details
        :return: list of output arguments : samples, mean inferred values, log-
            likelihood, log-evidence, error on log-evidence for each sample
        """
        mean_start, sigma_start = self._prepare_sampling(prior_type)

        if sampler_type == "dyPolyChord":
            if "resume_dyn_run" in kwargs_run and kwargs_run["resume_dyn_run"] is True:
                resume_dyn_run = True
            else:
                resume_dyn_run = False
            sampler = DyPolyChordSampler(
                self.likelihoodModule,
                prior_type=prior_type,
                prior_means=mean_start,
                prior_sigmas=sigma_start,
                width_scale=width_scale,
                sigma_scale=sigma_scale,
                output_dir=output_dir,
                output_basename=output_basename,
                polychord_settings=polychord_settings,
                remove_output_dir=remove_output_dir,
                resume_dyn_run=resume_dyn_run,
                use_mpi=self._mpi,
            )
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(
                dypolychord_dynamic_goal, kwargs_run
            )

        elif sampler_type == "MultiNest":
            sampler = MultiNestSampler(
                self.likelihoodModule,
                prior_type=prior_type,
                prior_means=mean_start,
                prior_sigmas=sigma_start,
                width_scale=width_scale,
                sigma_scale=sigma_scale,
                output_dir=output_dir,
                output_basename=output_basename,
                remove_output_dir=remove_output_dir,
                use_mpi=self._mpi,
            )
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(
                kwargs_run
            )
        else:
            sampler = DynestySampler(
                self.likelihoodModule,
                prior_type=prior_type,
                prior_means=mean_start,
                prior_sigmas=sigma_start,
                width_scale=width_scale,
                sigma_scale=sigma_scale,
                bound=dynesty_bound,
                sample=dynesty_sample,
                use_mpi=self._mpi,
            )
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(
                kwargs_run
            )

        # update current best fit values
        self._update_state(samples[-1])

        output = [
            sampler_type,
            samples,
            sampler.param_names,
            logL,
            logZ,
            logZ_err,
            results_object,
        ]

        return output

    def psf_iteration(self, compute_bands=None, **kwargs_psf_iter):
        """Iterative PSF reconstruction.

        :param compute_bands: bool list, if multiple bands, this process can be limited
            to a subset of bands
        :param kwargs_psf_iter: keyword arguments as used or available in
            PSFIteration.update_iterative() definition
        :return: 0, updated PSF is stored in self.multi_band_list
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_mask_list = kwargs_likelihood.get("image_likelihood_mask_list", None)
        kwargs_pixelbased = kwargs_likelihood.get("kwargs_pixelbased", None)
        kwargs_temp = self.best_fit(bijective=False)
        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for band_index in range(len(self.multi_band_list)):
            if compute_bands[band_index] is True:
                kwargs_psf = self.multi_band_list[band_index][1]
                image_model = SingleBandMultiModel(
                    self.multi_band_list,
                    kwargs_model,
                    likelihood_mask_list=likelihood_mask_list,
                    band_index=band_index,
                    kwargs_pixelbased=kwargs_pixelbased,
                )
                psf_iter = PsfFitting(image_model_class=image_model)
                kwargs_psf = psf_iter.update_iterative(
                    kwargs_psf, kwargs_params=kwargs_temp, **kwargs_psf_iter
                )
                self.multi_band_list[band_index][1] = kwargs_psf
        return 0

    def align_images(
        self,
        n_particles=10,
        n_iterations=10,
        align_offset=True,
        align_rotation=False,
        threadCount=1,
        compute_bands=None,
        delta_shift=0.2,
        delta_rot=0.1,
    ):
        """Aligns the coordinate systems of different exposures within a fixed model
        parameterisation by executing a PSO with relative coordinate shifts as free
        parameters.

        :param n_particles: number of particles in the Particle Swarm Optimization
        :param n_iterations: number of iterations in the optimization process
        :param align_offset: aligns shift in Ra and Dec
        :type align_offset: boolean
        :param align_rotation: aligns coordinate rotation
        :type align_rotation: boolean
        :param delta_shift: astrometric shift tolerance
        :param delta_rot: rotation angle tolerance [in radian]
        :param compute_bands: bool list, if multiple bands, this process can be limited
            to a subset of bands for which the coordinate system is being fit for best
            alignment to the model parameters
        :return: 0, updated coordinate system for the band(s)
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_mask_list = kwargs_likelihood.get("image_likelihood_mask_list", None)
        kwargs_temp = self.best_fit(bijective=False)
        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for i in range(len(self.multi_band_list)):
            if compute_bands[i] is True:
                alignmentFitting = AlignmentFitting(
                    self.multi_band_list,
                    kwargs_model,
                    kwargs_temp,
                    band_index=i,
                    likelihood_mask_list=likelihood_mask_list,
                    align_offset=align_offset,
                    align_rotation=align_rotation,
                )

                kwargs_data, chain = alignmentFitting.pso(
                    n_particles=n_particles,
                    n_iterations=n_iterations,
                    delta_shift=delta_shift,
                    delta_rot=delta_rot,
                    threadCount=threadCount,
                    mpi=self._mpi,
                    print_key="Alignment fitting for band %s ..." % i,
                )
                print("Align completed for band %s." % i)
                print(
                    "ra_shift: %s,  dec_shift: %s, phi_rot: %s"
                    % (
                        kwargs_data.get("ra_shift", 0),
                        kwargs_data.get("dec_shift", 0),
                        kwargs_data.get("phi_rot", 0),
                    )
                )
                self.multi_band_list[i][0] = kwargs_data
        return 0

    def flux_calibration(
        self,
        n_particles=10,
        n_iterations=10,
        threadCount=1,
        calibrate_bands=None,
        scaling_lower_limit=0,
        scaling_upper_limit=1000,
    ):
        """Calibrates flux_scaling between multiple images. This routine only works in
        'join-linear' model when fluxes are meant to be identical for different bands.

        :param n_particles: number of particles in the Particle Swarm Optimization
        :param n_iterations: number of iterations in the optimization process
        :param calibrate_bands: state which bands the flux calibration is applied to
        :type calibrate_bands: list of booleans of length of the imaging bands
        :param threadCount: number of CPU threads. If MPI option is set, threadCount=1
        :type threadCount: integer
        :param scaling_lower_limit: lower limit of flux_scaling
        :param scaling_upper_limit: upper limit of flux scaling
        :return: 0, updated coordinate system for the band(s)
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_temp = self.best_fit(bijective=False)
        multi_band_type = self.kwargs_data_joint.get("multi_band_type", "multi-linear")
        kwargs_imaging = self.likelihoodModule.kwargs_imaging

        calibration_fitting = FluxCalibration(
            kwargs_imaging=kwargs_imaging,
            kwargs_model=kwargs_model,
            kwargs_params=kwargs_temp,
            calibrate_bands=calibrate_bands,
        )

        multi_band_list, chain = calibration_fitting.pso(
            n_particles=n_particles,
            n_iterations=n_iterations,
            threadCount=threadCount,
            mpi=self._mpi,
            scaling_lower_limit=scaling_lower_limit,
            scaling_upper_limit=scaling_upper_limit,
        )
        self.multi_band_list = multi_band_list
        return 0

    def update_settings(
        self,
        kwargs_model=None,
        kwargs_constraints=None,
        kwargs_likelihood=None,
        lens_add_fixed=None,
        source_add_fixed=None,
        lens_light_add_fixed=None,
        ps_add_fixed=None,
        special_add_fixed=None,
        tracer_source_add_fixed=None,
        lens_remove_fixed=None,
        source_remove_fixed=None,
        lens_light_remove_fixed=None,
        ps_remove_fixed=None,
        special_remove_fixed=None,
        tracer_source_remove_fixed=None,
        change_source_lower_limit=None,
        change_source_upper_limit=None,
        change_lens_lower_limit=None,
        change_lens_upper_limit=None,
        change_sigma_lens=None,
        change_sigma_source=None,
        change_sigma_lens_light=None,
    ):
        """Updates lenstronomy settings "on the fly".

        :param kwargs_model: kwargs, specified keyword arguments overwrite the existing ones
        :param kwargs_constraints: kwargs, specified keyword arguments overwrite the existing ones
        :param kwargs_likelihood: kwargs, specified keyword arguments overwrite the existing ones
        :param lens_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param source_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_light_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param ps_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param special_add_fixed: ['param1', 'param2',...]
        :param special_add_fixed: ['param1', 'param2',...]
        :param tracer_source_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param source_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_light_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param ps_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param special_remove_fixed: ['param1', 'param2',...]
        :param special_remove_fixed: ['param1', 'param2',...]
        :param tracer_source_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param change_lens_lower_limit: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_lens_upper_limit: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_source_lower_limit: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_source_upper_limit: [[i_model, [''param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_sigma_lens: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_sigma_source: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :param change_sigma_lens_light: [[i_model, ['param_name1', 'param_name2', ...], [value1, value2, ...]]]
        :return: 0, the settings are overwritten for the next fitting step to come
        """
        self._updateManager.update_options(
            kwargs_model, kwargs_constraints, kwargs_likelihood
        )
        self._updateManager.update_fixed(
            lens_add_fixed,
            source_add_fixed,
            lens_light_add_fixed,
            ps_add_fixed,
            special_add_fixed,
            tracer_source_add_fixed,
            lens_remove_fixed,
            source_remove_fixed,
            lens_light_remove_fixed,
            ps_remove_fixed,
            special_remove_fixed,
            tracer_source_remove_fixed,
        )
        self._updateManager.update_limits(
            change_source_lower_limit,
            change_source_upper_limit,
            change_lens_lower_limit,
            change_lens_upper_limit,
        )
        self._updateManager.update_sigmas(
            change_sigma_lens=change_sigma_lens,
            change_sigma_source=change_sigma_source,
            change_sigma_lens_light=change_sigma_lens_light,
        )
        return 0

    def set_param_value(self, **kwargs):
        """Set a parameter to a specific value. `kwargs` are below.

        :param lens: [[i_model, ['param1', 'param2',...], [...]]
        :type lens:
        :param source: [[i_model, ['param1', 'param2',...], [...]]
        :type source:
        :param lens_light: [[i_model, ['param1', 'param2',...], [...]]
        :type lens_light:
        :param ps: [[i_model, ['param1', 'param2',...], [...]]
        :type ps:
        :return: 0, the value of the param is overwritten
        :rtype:
        """
        self._updateManager.update_param_value(**kwargs)

    def fix_not_computed(self, free_bands):
        """Fixes lens model parameters of imaging bands/frames that are not computed and
        frees the parameters of the other lens models to the initial kwargs_fixed
        options.

        :param free_bands: bool list of length of imaging bands in order of imaging
            bands, if False: set fixed lens model
        :return: None
        """
        self._updateManager.fix_not_computed(free_bands=free_bands)

    def _prepare_sampling(self, prior_type):
        if prior_type == "gaussian":
            mean_start = self.param_class.kwargs2args(
                **self._updateManager.parameter_state
            )
            sigma_start = self.param_class.kwargs2args(
                **self._updateManager.sigma_kwargs
            )
            mean_start = np.array(mean_start)
            sigma_start = np.array(sigma_start)
        else:
            mean_start, sigma_start = None, None
        return mean_start, sigma_start

    def _update_state(self, result):
        """

        :param result: array of parameters being sampled (e.g. result of MCMC chain)
        :return: None, updates the parameter state of the class instance
        """
        kwargs_result = self.param_class.args2kwargs(result, bijective=True)
        self._updateManager.update_param_state(**kwargs_result)

    def _result_from_mcmc(self, mcmc_output):
        """

        :param mcmc_output: list returned by self.mcmc()
        :return: kwargs_result like returned by self.pso(), from best logL MCMC sample
        """
        _, samples, _, logL_values = mcmc_output
        return self.best_fit_from_samples(samples, logL_values)

    def best_fit_from_samples(self, samples, logl):
        """Return best fit (max likelihood) value of samples in lenstronomy conventions.

        :param samples: samples of multi-dimensional parameter space
        :param logl: likelihood values for each sample
        :return: kwargs_result in lenstronomy convention
        """
        # get index of best logL sample
        best_fit_index = np.argmax(logl)
        best_fit_sample = samples[best_fit_index, :]
        best_fit_result = best_fit_sample.tolist()
        # get corresponding kwargs
        kwargs_result = self.param_class.args2kwargs(best_fit_result, bijective=True)
        return kwargs_result
