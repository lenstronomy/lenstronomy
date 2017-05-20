__author__ = 'sibirrer'

#this file is ment to be a shell script to be run with Monch cluster

# set up the scene
from cosmoHammer.util.MpiUtil import MpiPool
import time
import sys
import pickle
import os

from lenstronomy.Workflow.fitting_sequence import FittingSequence
pool = MpiPool(None)

start_time = time.time()

job_name = str(sys.argv[1])
if pool.isMaster():
    print("job %s loaded" %job_name)
# hoffman2 specifics
dir_path_cluster = '/u/scratch/s/sibirrer/'
path2load = os.path.join(dir_path_cluster, job_name)+".txt"
path2dump = os.path.join(dir_path_cluster, job_name)+"_out.txt"

f = open(path2load, 'rb')
input = pickle.load(f)
f.close()
[kwargs_data, kwargs_psf, kwargs_options, kwargs_init, kwargs_sigma, kwargs_fixed, fitting_kwargs_list, init_samples] = input
fitting_seq = FittingSequence(kwargs_data, kwargs_psf, kwargs_options, kwargs_init, kwargs_sigma, kwargs_fixed)

lens_result, source_result, lens_light_result, else_result, chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list)

# save the output
if pool.isMaster():
    f = open(path2dump, 'wb')
    output = [lens_result, source_result, lens_light_result, else_result, chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc]
    pickle.dump([input, output], f)
    f.close()
    end_time = time.time()
    print(end_time - start_time, 'total time needed for computation')
    print('Result saved in: %s' % path2dump)
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')