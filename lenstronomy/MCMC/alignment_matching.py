__author__ = 'sibirrer'

import time
import copy
from cosmoHammer import MpiParticleSwarmOptimizer
from cosmoHammer import ParticleSwarmOptimizer
from cosmoHammer.util import MpiUtil
import lenstronomy.Util.class_creator as class_creator


class AlignmentFitting(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, compute_bool=None):
        """
        initialise the classes of the chain and for parameter options
        """
        self.chain = AlignmentChain(kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, compute_bool=compute_bool)

    def pso(self, n_particles, n_iterations, lowerLimit, upperLimit, threadCount=1, mpi=False, print_key='default'):
        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """
        init_pos = self.chain.get_args(self.chain._kwargs_data_init)
        num_param = self.chain.num_param
        lowerLimit = [lowerLimit] * num_param
        upperLimit = [upperLimit] * num_param
        if mpi is True:
            pso = MpiParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=1)
        else:
            pso = ParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=threadCount)
        if not init_pos is None:
            pso.gbest.position = init_pos
            pso.gbest.velocity = [0]*len(init_pos)
            pso.gbest.fitness, _ = self.chain.likelihood(init_pos)
        X2_list = []
        vel_list = []
        pos_list = []
        time_start = time.time()
        if pso.isMaster():
            print('Computing the %s ...' % print_key)
        num_iter = 0
        for swarm in pso.sample(n_iterations):
            X2_list.append(pso.gbest.fitness*2)
            vel_list.append(pso.gbest.velocity)
            pos_list.append(pso.gbest.position)
            num_iter += 1
            if pso.isMaster():
                if num_iter % 10 == 0:
                    print(num_iter)
        if not mpi:
            result = pso.gbest.position
        else:
            result = MpiUtil.mpiBCast(pso.gbest.position)
        kwargs_data = self.chain.update_data(result)
        if mpi is True and not pso.isMaster():
            pass
        else:
            time_end = time.time()
            print("Shifts found: ", result)
            print(time_end - time_start, 'time used for PSO', print_key)
        return kwargs_data, [X2_list, pos_list, vel_list, []]


class AlignmentChain(object):

    def __init__(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, compute_bool=None):
        """
        initializes all the classes needed for the chain
        """
        # print('initialized on cpu', threading.current_thread())

        self._kwargs_data_init = kwargs_data
        self._kwargs_psf = kwargs_psf
        self._compute_bool = compute_bool
        self._source_marg = kwargs_options.get('source_marg', False)
        kwargs_options_copy = copy.deepcopy(kwargs_options)
        kwargs_options_copy['error_map'] = False
        self._kwargs_options = kwargs_options_copy
        self._kwargs_lens, self._kwargs_source, self._kwargs_lens_light, self._kwargs_else = kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else

    def X2_chain_image(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #generate image and computes likelihood
        kwargs_data = self.update_data(args)
        makeImageMultiband = class_creator.creat_multiband(kwargs_data, self._kwargs_psf, self._kwargs_options, compute_bool=self._compute_bool)
        logL = makeImageMultiband.likelihood_data_given_model(self._kwargs_lens, self._kwargs_source, self._kwargs_lens_light, self._kwargs_else, source_marg=self._source_marg)
        return logL, None

    def __call__(self, a):
        return self.X2_chain_image(a)

    def likelihood(self, a):
        return self.X2_chain_image(a)

    def computeLikelihood(self, ctx):
        logL, _ = self.X2_chain_image(ctx.getParams())
        return logL

    def setup(self):
        pass

    def update_data(self, args):
        """

        :param args:
        :return:
        """
        k = 0
        kwargs_data = copy.deepcopy(self._kwargs_data_init)
        for i, kwargs_data_i in enumerate(kwargs_data):
            if self._compute_bool[i]:
                kwargs_data_i['ra_shift'] = args[k]
                kwargs_data_i['dec_shift'] = args[k+1]
                k += 2
        return kwargs_data

    def get_args(self, kwargs_data):
        """

        :param kwargs_data:
        :return:
        """
        args = []
        for i, kwargs_data_i in enumerate(kwargs_data):
            if self._compute_bool[i]:
                args.append(kwargs_data_i.get('ra_shift', 0))
                args.append(kwargs_data_i.get('dec_shift', 0))
        return args

    @property
    def num_param(self):
        n = 0
        for i in range(len(self._kwargs_data_init)):
            if self._compute_bool[i]:
                n += 2
        return n
