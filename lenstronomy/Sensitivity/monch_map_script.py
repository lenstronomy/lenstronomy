__author__ = 'sibirrer'

#this file is ment to be a shell script to be run with Monch cluster

# set up the scene


import time
import sys
import pickle
import os

start_time = time.time()

path2load = '/mnt/lnec/sibirrer/' + str(sys.argv[1]) +".txt"
num_jobs = int(sys.argv[2])
print("file to be loaded: %s" %path2load)
f = open(path2load, 'rb')
[kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                          kwargs_lens_light, kwargs_else, phi_E_clump, r_trunc, x_clump, y_clump, kwargs_compute, path2dump] = pickle.load(f)
f.close()
os.system("cd /users/sibirrer/Lenstronomy/lenstronomy/Sensitivity/")

num_calc = len(x_clump)
n_iter = 0
step = int(num_calc/num_jobs+0.5)
for i in range(num_jobs):
    if i < num_jobs - 1:
        x_clump_i = x_clump[n_iter:n_iter+step]
        y_clump_i = y_clump[n_iter:n_iter + step]
        n_iter += step
    else:
        x_clump_i = x_clump[n_iter:]
        y_clump_i = x_clump[n_iter:]
    path2input_i = '/mnt/lnec/sibirrer/' + path2dump+"_"+str(i) +".txt"
    path2output_i = '/mnt/lnec/sibirrer/' + path2dump+"_"+str(i)+"_out" +".txt"
    f = open(path2input_i, 'wb')
    pickle.dump([kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                              kwargs_lens_light, kwargs_else, phi_E_clump, r_trunc, x_clump_i, y_clump_i, kwargs_compute, path2output_i], f)
    f.close()
    os.system("sbatch map_script.sh "+path2input_i)


print("All jobs submitted, outputs will be stored in %s ." %(path2dump+"_i_out"+".txt"))