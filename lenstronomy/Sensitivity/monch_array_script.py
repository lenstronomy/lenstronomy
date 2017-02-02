__author__ = 'sibirrer'

#this file is ment to be a shell script to be run with Monch cluster

# set up the scene


import time
import sys
import pickle
import numpy as np

from lenstronomy.Sensitivity.sensitivity_map import SensitivityMap
from lenstronomy.Sensitivity.clump_detection import ClumpDetect

start_time = time.time()

path2load = '/mnt/lnec/sibirrer/' + str(sys.argv[1]) +".txt"
i = int(sys.argv[2])
num_jobs = int(sys.argv[3])
print("file to be loaded: %s" %path2load)
f = open(path2load, 'rb')
[kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                          kwargs_lens_light, kwargs_else, phi_E_clump, r_trunc, x_clump, y_clump, kwargs_compute, path2dump] = pickle.load(f)
f.close()

compute_bool = kwargs_compute.get("compute_bool", np.ones_like(x_clump))

num_calc = len(x_clump)
num_compute = np.sum(compute_bool)
step_new = int(num_compute/float(num_jobs)+0.5)

i_min = i*step_new
if i == num_jobs - 1:
    i_max = num_compute+1
else:
    i_max = (i+1)*step_new

x_clump_i_new = []
y_clump_i_new = []
compute_bool_i = []
counter = 0
for k in range(num_calc):
    if counter >= i_min and counter < i_max:
        x_clump_i_new.append(x_clump[k])
        y_clump_i_new.append(y_clump[k])
        compute_bool_i.append(compute_bool[k])
    if compute_bool[k] == 1:
        counter += 1


print("start job", i)

"""


step = int(num_calc/num_jobs+0.5)
n_iter = i * step
if i < num_jobs - 1:
    x_clump_i = x_clump[n_iter:n_iter+step]
    y_clump_i = y_clump[n_iter:n_iter + step]
    compute_bool_i = compute_bool[n_iter:n_iter + step]
    n_iter += step
else:
    x_clump_i = x_clump[n_iter:]
    y_clump_i = x_clump[n_iter:]
    compute_bool_i = compute_bool[n_iter:]
"""

path2output_i = '/mnt/lnec/sibirrer/' + path2dump+"_"+str(i)+"_out" +".txt"


if kwargs_compute['detect_only'] is False:
    sensitivityMap = SensitivityMap(kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                              kwargs_lens_light, kwargs_else)
    chi2_list_smooth_data, chi2_list_clump_data, chi2_list_smooth_sens, chi2_list_clump_sens = sensitivityMap.iterate_position(phi_E_clump, r_trunc, x_clump_i_new, y_clump_i_new, compute_bool_i)
else:
    clumpDetection = ClumpDetect(kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                              kwargs_lens_light, kwargs_else)
    chi2_list_smooth_data, chi2_list_clump_data = clumpDetection.iterate_position(phi_E_clump, r_trunc, x_clump_i_new, y_clump_i_new, compute_bool_i)
    chi2_list_smooth_sens, chi2_list_clump_sens = np.zeros_like(chi2_list_smooth_data), np.zeros_like(chi2_list_clump_data)


# save the output
f = open(path2output_i, 'wb')
pickle.dump([chi2_list_smooth_data, chi2_list_clump_data, chi2_list_smooth_sens, chi2_list_clump_sens, phi_E_clump, r_trunc, x_clump_i_new, y_clump_i_new],f)
f.close()

end_time = time.time()
print(end_time - start_time, 'total time needed for computation')
print('Result saved in:', path2output_i)
print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')