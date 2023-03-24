__author__ = 'sibirrer'

import time
import copy
from lenstronomy.Sampling.Pool.pool import choose_pool
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer

__all__ = ['CalibrationFitting', 'CalibrationLikelihood']


class CalibrationFitting(object):
    """
    class to fit coordinate system alignment and flux amplitude calibrations
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_params, band_index=0, likelihood_mask_list=None,
                 image_alignment=True, flux_calibration=False):
        """
        initialise the classes of the chain and for parameter options
        """
        self._flux_calibration = flux_calibration
        self._image_alignment = image_alignment
        self.chain = CalibrationLikelihood(multi_band_list, kwargs_model, kwargs_params, band_index,
                                           likelihood_mask_list, image_alignment=image_alignment,
                                           flux_calibration=flux_calibration)

    def pso(self, n_particles=10, n_iterations=10, threadCount=1, mpi=False, kwargs_lower_limit=None,
            kwargs_upper_limit=None,
            print_key='alignment or flux calibration'):
        """
        returns the best fit for the lens model on catalogue basis with particle swarm optimizer

        """
        init_pos = self.chain.get_args(self.chain.kwargs_data_init)
        if kwargs_upper_limit is None:
            kwargs_upper_limit = {}
        if kwargs_lower_limit is None:
            kwargs_lower_limit = {}
        # if not provided, use defaults
        kwargs_lower_limit_update = {**{'ra_shift': -0.2, 'dec_shift': -0.2, 'flux_scaling': 0}, **kwargs_lower_limit}
        kwargs_upper_limit_update = {**{'ra_shift': 0.2, 'dec_shift': 0.2, 'flux_scaling': 10}, **kwargs_upper_limit}
        print(kwargs_lower_limit_update, 'test kwargs_lower_limit')
        lower_limit = self.chain.get_args(kwargs_lower_limit_update)
        upper_limit = self.chain.get_args(kwargs_upper_limit_update)

        pool = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)

        pso = ParticleSwarmOptimizer(self.chain, lower_limit, upper_limit,
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
            print("parameters found: ", result)
            print(time_end - time_start, 'time used for ', print_key)
        return kwargs_data, [chi2_list, pos_list, vel_list]


class CalibrationLikelihood(object):

    def __init__(self, multi_band_list, kwargs_model, kwargs_params, band_index=0, likelihood_mask_list=None,
                 image_alignment=True, flux_calibration=False):
        """
        initializes all the classes needed for the chain

        :param multi_band_list:
        :param kwargs_model:
        :param kwargs_params:
        :param band_index:
        :param likelihood_mask_list:
        :param image_alignment:
        :param flux_calibration:
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
        self._alignment_matching = image_alignment
        self._flux_calibration = flux_calibration

    def _likelihood(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        # generate image and computes likelihood
        multi_band_list = self.update_multi_band(args)
        image_model = SingleBandMultiModel(multi_band_list, self._kwargs_model,
                                           likelihood_mask_list=self._likelihood_mask_list, band_index=self._band_index)
        log_likelihood = image_model.likelihood_data_given_model(source_marg=self._source_marg, **self._kwargs_params)
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
        if self._alignment_matching:
            kwargs_data['ra_shift'] = args[k]
            kwargs_data['dec_shift'] = args[k+1]
            k += 2
        if self._flux_calibration:
            kwargs_data['flux_scaling'] = args[k]
        return kwargs_data

    def get_args(self, kwargs_data):
        """

        :param kwargs_data:
        :return:
        """
        args = []
        if self._alignment_matching:
            args.append(kwargs_data.get('ra_shift', 0))
            args.append(kwargs_data.get('dec_shift', 0))
        if self._flux_calibration:
            args.append(kwargs_data.get('flux_scaling', 1))
        return args

    @property
    def num_param(self):
        n = 0
        if self._alignment_matching:
            n += 2
        if self._flux_calibration:
            n += 1
        return n
