__author__ = "sibirrer"

import time
import copy
from lenstronomy.Sampling.Pool.pool import choose_pool
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer

__all__ = ["AlignmentFitting", "AlignmentLikelihood"]


class AlignmentFitting(object):
    """Class which executes the different sampling  methods."""

    def __init__(
        self,
        multi_band_list,
        kwargs_model,
        kwargs_params,
        band_index=0,
        likelihood_mask_list=None,
        align_offset=True,
        align_rotation=False,
    ):
        """Initialise the classes of the chain and for parameter options.

        :param align_offset: aligns shift in Ra and Dec
        :type align_offset: boolean
        :param align_rotation: aligns coordinate rotation
        :type align_rotation: boolean
        """
        self.chain = AlignmentLikelihood(
            multi_band_list,
            kwargs_model,
            kwargs_params,
            band_index,
            likelihood_mask_list,
            align_rotation=align_rotation,
            align_offset=align_offset,
        )

    def pso(
        self,
        n_particles=10,
        n_iterations=10,
        delta_shift=0.2,
        delta_rot=0.1,
        threadCount=1,
        mpi=False,
        print_key="default",
    ):
        """Returns the best fit for the lens model on catalogue basis with particle
        swarm optimizer.

        :param n_particles:
        :param n_iterations:
        :param delta_shift: astrometric shift tolerance
        :param delta_rot: rotation angle tolerance [in radian]
        :param threadCount:
        :param mpi:
        :param print_key:
        :return:
        """
        init_pos = self.chain.get_args(self.chain.kwargs_data_init)
        lower_limit, upper_limit = self.chain.lower_upper_limit(delta_shift, delta_rot)
        pool = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)

        pso = ParticleSwarmOptimizer(
            self.chain, lower_limit, upper_limit, n_particles, pool=pool
        )
        if init_pos is not None:
            pso.set_global_best(
                init_pos, [0] * len(init_pos), self.chain.likelihood(init_pos)
            )

        if pool.is_master():
            print("Computing the %s ..." % print_key)

        time_start = time.time()

        result, [chi2_list, pos_list, vel_list] = pso.optimize(n_iterations)

        kwargs_data = self.chain.update_data(result)

        if pool.is_master():
            time_end = time.time()
            print("Shifts found: ", result)
            print(time_end - time_start, "time used for ", print_key)
        return kwargs_data, [chi2_list, pos_list, vel_list]


class AlignmentLikelihood(object):
    def __init__(
        self,
        multi_band_list,
        kwargs_model,
        kwargs_params,
        band_index=0,
        likelihood_mask_list=None,
        align_offset=True,
        align_rotation=False,
    ):
        """Initializes all the classes needed for the chain.

        :param align_offset: aligns shift in Ra and Dec
        :type align_offset: boolean
        :param align_rotation: aligns coordinate rotation
        :type align_rotation: boolean
        """
        # print('initialized on cpu', threading.current_thread())
        self._align_offset = align_offset
        self._align_rotation = align_rotation
        self._multi_band_list = multi_band_list
        self.kwargs_data_init = multi_band_list[band_index][0]
        self._kwargs_data_shifted = copy.deepcopy(self.kwargs_data_init)

        self._kwargs_model = kwargs_model
        self._source_marg = False
        self._band_index = band_index
        self._likelihood_mask_list = likelihood_mask_list
        self._kwargs_params = kwargs_params

    def _likelihood(self, args):
        """Routine to compute X2 given variable parameters for a MCMC/PSO chainF."""
        # generate image and computes likelihood
        multi_band_list = self.update_multi_band(args)
        image_model = SingleBandMultiModel(
            multi_band_list,
            self._kwargs_model,
            likelihood_mask_list=self._likelihood_mask_list,
            band_index=self._band_index,
        )
        log_likelihood = image_model.likelihood_data_given_model(
            source_marg=self._source_marg, **self._kwargs_params
        )
        return log_likelihood

    def __call__(self, a):
        return self._likelihood(a)

    def likelihood(self, a):
        return self._likelihood(a)

    def setup(self):
        pass

    def update_multi_band(self, args):
        """
        :param args: list of parameters
        :return: updated multi_band_list
        """
        kwargs_data = self.update_data(args)
        multi_band_list = self._multi_band_list
        multi_band_list[self._band_index][0] = kwargs_data
        return multi_band_list

    def update_data(self, args):
        """
        :param args:
        :return:
        """
        k = 0
        kwargs_data = self._kwargs_data_shifted
        if self._align_offset:
            kwargs_data["ra_shift"] = args[k]
            kwargs_data["dec_shift"] = args[k + 1]
            k += 2
        if self._align_rotation:
            kwargs_data["phi_rot"] = args[k]
            k += 1
        return kwargs_data

    def get_args(self, kwargs_data):
        """
        :param kwargs_data: keyword arguments for ImageData()
        :return: arguments being sampled
        """
        args = []
        if self._align_offset:
            args.append(kwargs_data.get("ra_shift", 0))
            args.append(kwargs_data.get("dec_shift", 0))
        if self._align_rotation:
            args.append(kwargs_data.get("phi_rot", 0))
        return args

    @property
    def num_param(self):
        n = 0
        if self._align_offset:
            n += 2
        if self._align_rotation:
            n += 1
        return n

    def lower_upper_limit(self, delta_shift, delta_rot):
        """

        :param delta_shift: astrometric shift tolerance
        :param delta_rot: rotation angle tolerance [in radian]
        :return: lower_limit, upper_limit
        """
        lower_limit, upper_limit = [], []
        if self._align_offset:
            lower_limit.append(-delta_shift)
            lower_limit.append(-delta_shift)
            upper_limit.append(delta_shift)
            upper_limit.append(delta_shift)
        if self._align_rotation:
            lower_limit.append(-delta_rot)
            upper_limit.append(delta_rot)
        return lower_limit, upper_limit
