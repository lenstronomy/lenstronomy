__author__ = 'sibirrer'

import time
import copy
from lenstronomy.Sampling.Pool.pool import choose_pool
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer

__all__ = ['AlignmentFitting', 'AlignmentLikelihood']


class AlignmentFitting(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_params, band_index=0, likelihood_mask_list=None):
        """
        initialise the classes of the chain and for parameter options
        """
        self.chain = AlignmentLikelihood(multi_band_list, kwargs_model, kwargs_params, band_index, likelihood_mask_list)

    def pso(self, n_particles=10, n_iterations=10, lowerLimit=-0.2, upperLimit=0.2, threadCount=1, mpi=False,
            print_key='default'):
        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """
        init_pos = self.chain.get_args(self.chain.kwargs_data_init)
        num_param = self.chain.num_param
        lowerLimit = [lowerLimit] * num_param
        upperLimit = [upperLimit] * num_param

        pool = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)

        pso = ParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit,
                                     n_particles, pool=pool)
        if init_pos is not None:
            pso.set_global_best(init_pos, [0]*len(init_pos),
                                self.chain.likelihood(init_pos))

        if pool.is_master():
            print('Computing the %s ...' % print_key)

        time_start = time.time()

        result, [chi2_list, pos_list, vel_list] = pso.optimize(n_iterations)

        kwargs_data = self.chain.update_data(result)

        if pool.is_master():
            time_end = time.time()
            print("Shifts found: ", result)
            print(time_end - time_start, 'time used for ', print_key)
        return kwargs_data, [chi2_list, pos_list, vel_list]


class AlignmentLikelihood(object):

    def __init__(self, multi_band_list, kwargs_model, kwargs_params, band_index=0, likelihood_mask_list=None):
        """
        initializes all the classes needed for the chain
        """
        # print('initialized on cpu', threading.current_thread())
        self._multi_band_list = multi_band_list
        self.kwargs_data_init = multi_band_list[band_index][0]
        self._kwargs_data_shifted = copy.deepcopy(self.kwargs_data_init)

        self._kwargs_model = kwargs_model
        self._source_marg = False
        self._band_index = band_index
        self._likelihood_mask_list = likelihood_mask_list
        self._kwargs_params = kwargs_params

    def _likelihood(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        # generate image and computes likelihood
        multi_band_list = self.update_multi_band(args)
        image_model = SingleBandMultiModel(multi_band_list, self._kwargs_model,
                                           likelihood_mask_list=self._likelihood_mask_list, band_index=self._band_index)
        logL = image_model.likelihood_data_given_model(source_marg=self._source_marg, **self._kwargs_params)
        return logL

    def __call__(self, a):
        return self._likelihood(a)

    def likelihood(self, a):
        return self._likelihood(a)

    def computeLikelihood(self, ctx):
        logL, _ = self._likelihood(ctx.args2kwargs())
        return logL

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
        kwargs_data['ra_shift'] = args[k]
        kwargs_data['dec_shift'] = args[k+1]
        k += 2
        return kwargs_data

    @staticmethod
    def get_args(kwargs_data):
        """

        :param kwargs_data:
        :return:
        """
        args = []
        args.append(kwargs_data.get('ra_shift', 0))
        args.append(kwargs_data.get('dec_shift', 0))
        return args

    @property
    def num_param(self):
        n = 2
        return n
