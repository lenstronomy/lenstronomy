__author__ = 'sibirrer'

#this file is ment to be a shell script to be run with Monch cluster

# set up the scene
import os
import pickle
import sys
import time

from cosmoHammer.util.MpiUtil import MpiPool
from lenstronomy.Trash.fitting import Fitting

start_time = time.time()

job_name = str(sys.argv[1])
# hoffman2 specifics
dir_path_cluster = '/u/scratch/s/sibirrer/'
path2load = os.path.join(dir_path_cluster, job_name)+".txt"
path2dump = os.path.join(dir_path_cluster, job_name)+"_out.txt"

f = open(path2load, 'rb')
[kwargs_data, kwargs_psf, kwargs_options, kwargs_mcmc, kwargs_mean, kwargs_sigma, kwargs_fixed, init_samples] = pickle.load(f)
f.close()

kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = kwargs_mean
kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma = kwargs_sigma
kwargs_lens_fixed, kwargs_source_fixed, kwargs_lens_light_fix, kwargs_else_fixed = kwargs_fixed

fitting = Fitting(kwargs_data, kwargs_psf, kwargs_lens_fixed, kwargs_source_fixed, kwargs_lens_light_fix, kwargs_else_fixed)

n_burn = kwargs_mcmc['n_burn']
n_run = kwargs_mcmc['n_run']
walkerRatio = kwargs_mcmc['walkerRatio']
mpi = kwargs_mcmc['mpi']


samples, params, dist = fitting.mcmc_run(kwargs_options, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else,
                 kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_else_sigma,
                 n_burn, n_run, walkerRatio, threadCount=1, mpi=mpi, init_samples=init_samples)

# save the output
pool = MpiPool(None)
if pool.isMaster():
    f = open(path2dump, 'wb')
    pickle.dump([samples, params, dist], f)
    f.close()
    end_time = time.time()
    print(end_time - start_time, 'total time needed for computation')
    print('Result saved in: %s' % path2dump)
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')