__author__ = 'sibirrer'

import os
import pickle
import sys
import time

start_time = time.time()

path2load = str(sys.argv[1])
f = open(path2load, 'rb')
[kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                          kwargs_lens_light, kwargs_else, phi_E_clump, r_trunc, x_clump, y_clump, kwargs_compute, path2dump] = pickle.load(f)
f.close()

os.system("rm "+path2load)
print("file loaded, computation is starting...")

from lenstronomy.Extensions.Substructure.sensitivity_map import SensitivityMap
from lenstronomy.Extensions.Substructure.clump_detection import ClumpDetect
if kwargs_compute['detect_only'] is False:
    sensitivityMap = SensitivityMap(kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                              kwargs_lens_light, kwargs_else)
    chi2_list_smooth_data, chi2_list_clump_data, chi2_list_smooth_sens, chi2_list_clump_sens = sensitivityMap.iterate_position(phi_E_clump, r_trunc, x_clump, y_clump)
else:
    clumpDetection = ClumpDetect(kwargs_options, kwargs_data, kwargs_lens, kwargs_source, kwargs_psf,
                              kwargs_lens_light, kwargs_else)
    chi2_list_smooth_data, chi2_list_clump_data = clumpDetection.iterate_position(phi_E_clump, r_trunc, x_clump, y_clump)
    chi2_list_smooth_sens, chi2_list_clump_sens = 0, 0


# save the output
f = open(path2dump, 'wb')
pickle.dump([chi2_list_smooth_data, chi2_list_clump_data, chi2_list_smooth_sens, chi2_list_clump_sens, phi_E_clump, r_trunc, x_clump, y_clump],f)
f.close()

end_time = time.time()
print(end_time - start_time, 'total time needed for computation')
print('Result saved in:', path2dump)
print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')
