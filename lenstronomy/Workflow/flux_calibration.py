__author__ = 'sibirrer'

import time
import copy
from lenstronomy.Sampling.Pool.pool import choose_pool
from lenstronomy.Sampling.Likelihoods.image_likelihood import ImageLikelihood
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer

__all__ = ['FluxCalibration', 'CalibrationLikelihood']


class FluxCalibration(object):
    """
    class to fit coordinate system alignment and flux amplitude calibrations
    """
    def __init__(self, kwargs_imaging, kwargs_model, kwargs_params, calibrate_bands):
        """
        initialise the classes of the chain and for parameter options
        """
        multi_band_list = kwargs_imaging['multi_band_list']
        multi_band_type = kwargs_imaging['multi_band_type']

        if calibrate_bands is None:
            calibrate_bands = [False] * len(multi_band_list)
        if multi_band_type != 'joint-linear':
            raise ValueError('flux calibration should only be done with join-linear data model!')
        self._calibrate_bands = calibrate_bands
        self.chain = CalibrationLikelihood(kwargs_model, kwargs_params,
                                           calibrate_bands=calibrate_bands,
                                           kwargs_imaging=kwargs_imaging)

    def pso(self, n_particles=10, n_iterations=10, threadCount=1, mpi=False, scaling_lower_limit=0,
            scaling_upper_limit=1000, print_key='flux calibration'):
        """
        returns the best fit for the lens model on catalogue basis with particle swarm optimizer

        """
        init_pos = self.chain.get_args(self.chain.multi_band_list)
        pool = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)
        num_param = self.chain.num_param
        lower_limit = [scaling_lower_limit] * num_param
        upper_limit = [scaling_upper_limit] * num_param

        pso = ParticleSwarmOptimizer(self.chain, lower_limit, upper_limit, n_particles, pool=pool)
        pso.set_global_best(init_pos, [0]*len(init_pos), self.chain.likelihood(init_pos))

        if pool.is_master():
            print('Computing the %s ...' % print_key)

        time_start = time.time()

        result, [chi2_list, pos_list, vel_list] = pso.optimize(n_iterations)

        multi_band_list = self.chain.update_data(result)

        if pool.is_master():
            time_end = time.time()
            print("parameters found: ", result)
            print(time_end - time_start, 'time used for ', print_key)
            print('Calibration completed for bands %s.' % self._calibrate_bands)
        return multi_band_list, [chi2_list, pos_list, vel_list]


class CalibrationLikelihood(object):

    def __init__(self, kwargs_model, kwargs_params, calibrate_bands, kwargs_imaging):
        """
        initializes all the classes needed for the chain

        :param multi_band_list:
        :param kwargs_model:
        :param kwargs_params:
        :param calibrate_bands: state which bands the flux calibration is applied to
        :type calibrate_bands: list of booleans of length of the imaging bands
        :param kwargs_imaging: keyword arguments of the imaging likelihood
        """
        self._calibrate_bands = calibrate_bands
        self._kwargs_model = kwargs_model
        self._kwargs_params = kwargs_params
        self._kwargs_imaging_likelihood = copy.deepcopy(kwargs_imaging)
        self.multi_band_list = self._kwargs_imaging_likelihood['multi_band_list']

    def _likelihood(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        # generate image and computes likelihood
        multi_band_list = self.update_data(args)
        self._kwargs_imaging_likelihood['multi_band_list'] = multi_band_list
        # this line is redundant since the self.multi_band_list variable got already updated
        image_likelihood = ImageLikelihood(kwargs_model=self._kwargs_model, **self._kwargs_imaging_likelihood)
        log_likelihood = image_likelihood.logL(**self._kwargs_params)
        return log_likelihood

    def __call__(self, a):
        return self._likelihood(a)

    def likelihood(self, a):
        return self._likelihood(a)

    def setup(self):
        pass

    def update_data(self, args):
        """

        :param args:
        :return: updated multi_band_list
        """
        k = 0
        for i, band in enumerate(self.multi_band_list):
            if self._calibrate_bands[i]:
                kwargs_data = band[0]
                kwargs_data['flux_scaling'] = args[k]
                k += 1
        return self.multi_band_list

    def get_args(self, multi_band_list):
        """

        :param multi_band_list: list of multi_band [[kwargs_data, kwargs_psf, kwargs_numeric], [...], ...]
        :return:
        """
        args = []
        for i, band in enumerate(multi_band_list):
            if self._calibrate_bands[i]:
                args.append(band[0].get('flux_scaling', 1))
        return args

    @property
    def num_param(self):
        n = 0
        for i in range(len(self._calibrate_bands)):
            if self._calibrate_bands[i]:
                n += 1
        return n
