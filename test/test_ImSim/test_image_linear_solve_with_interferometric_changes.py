import numpy as np
import numpy.testing as npt
import scipy.signal
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Util import kernel_util
import lenstronomy.Util.util as util

from lenstronomy.ImSim.image_linear_solve import ImageLinearFit

"""
Test the linear solver for natwt (natural weighting) interferometric data.
Test the _image_linear_solve function of ImageLinearFit class.
The idea is to define data, psf, source, lens, lens light classes respectively, and run the linear solving
inside and outside of the _image_linear_solve function. Verify the 1st and 4th output of _image_linear_solve.
The test should be independent of the specific definitions of the light and lens profiles.
"""


def test_image_linear_solve_with_primary_beam_and_interferometry_psf():
    
    background_rms = .05 
    exp_time = np.inf 
    numPix = 80 
    deltaPix = 0.05  
    psf_type = 'PIXEL'  
    kernel_size = 161 
    
    # simulate a primary beam (pb)
    primary_beam = np.zeros((numPix,numPix))
    for i in range(numPix):
        for j in range(numPix):
            primary_beam[i,j] = np.exp(-1e-4*((i-78)**2+(j-56)**2))
    primary_beam /= np.max(primary_beam)
    
    # simulate a spherical sinc function as psf, which contains negative pixels
    psf_test = np.zeros((221,221))
    for i in range(221):
        for j in range(221):
            if i > j:
                psf_test[i,j] = psf_test[j,i]
            r = np.sqrt((i-110)**2 + (j-110)**2)
            if r == 0:
                psf_test[i,j] = 1
            else:
                psf_test[i,j] = np.sin(r*0.5)/(r*0.5)
    
    # note that the simulated noise here is not the interferometric noise. we just use it to test the numerics
    test_noise = scipy.signal.fftconvolve(np.random.normal(0,1,(numPix,numPix)),psf_test,mode='same')
    
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
    kwargs_data['ra_at_xy_0'] = -(40)*deltaPix
    kwargs_data['dec_at_xy_0'] = -(40)*deltaPix 
    kwargs_data['antenna_primary_beam'] = primary_beam
    kwargs_data['likelihood_method'] = 'interferometry_natwt' # testing just for interferometry natwt method
    data_class = ImageData(**kwargs_data)
    
    kernel_cut = kernel_util.cut_psf(psf_test, kernel_size, normalization = False)
    kwargs_psf = {'psf_type': psf_type,'pixel_size': deltaPix, 'kernel_point_source': kernel_cut,'kernel_point_source_normalization': False}
    psf_class = PSF(**kwargs_psf)
    
    # define lens model and source model
    kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}
    kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.04}
    lens_model_list = ['SPEP', 'SHEAR']
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    
    kwargs_sersic = {'amp': 25., 'R_sersic': 0.3, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
    lens_light_model_list = ['SERSIC']
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    
    kwargs_sersic_ellipse = {'amp': 10., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                             'e1': 0.05, 'e2': 0.02}
    source_model_list = ['SERSIC_ELLIPSE']
    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)
    
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, kwargs_numerics=kwargs_numerics)
    image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light)
    
    # normalize the noise to make it small compared to the model image
    test_noise *= 1e-2 * (np.max(image_sim) / np.std(test_noise))
    sim_data = image_sim + test_noise
    data_class.update_data(sim_data)
    
    # define the ImageLinearFit class using the materials defined above, run the _image_linear_solve function
    imageLinearFit = ImageLinearFit(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, kwargs_numerics=kwargs_numerics)
    model,_,_,amps = imageLinearFit._image_linear_solve(kwargs_lens, kwargs_source, kwargs_lens_light)
    
    # execute the same linear solving outside of the _image_linear_solve function
    A = imageLinearFit._linear_response_matrix(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = None, unconvolved=True)
    A0 = util.array2image(A[0])
    A1 = util.array2image(A[1])
    A0c = scipy.signal.fftconvolve(A0, psf_test, mode = 'same')
    A1c = scipy.signal.fftconvolve(A1, psf_test, mode = 'same')
    M = np.zeros((2,2))
    b = np.zeros((2))
    M[0,0] = np.sum(A0c * A0)
    M[0,1] = np.sum(A0c * A1)
    M[1,0] = np.sum(A1c * A0)
    M[1,1] = np.sum(A1c * A1)
    b[0] = np.sum(A0 * sim_data)
    b[1] = np.sum(A1 * sim_data)
    
    amps0 = np.linalg.lstsq(M, b)[0]
    clean_model = amps0[0] * A0 + amps0[1] * A1
    dirty_model = amps0[0] * A0c + amps0[1] * A1c
    
    npt.assert_almost_equal([clean_model, dirty_model], model, decimal=8)
    npt.assert_almost_equal(amps0, amps, decimal=8)