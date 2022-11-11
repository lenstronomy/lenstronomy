import numpy.testing as npt
import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

"""
test the antenna primary beam (maybe be modified further to test all interferometric changes)
The idea of this test is to define two data classes, one with the antenna primary beam, one without, 
and compare the (image_with_pb) with (image_no_pb * pb). ('pb' is short for primary beam.)
"""

def test_interferometric_changes():

    sigma_bkg = .05  # background noise per pixel
    exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    numPix = 100  # cutout pixel size
    deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
    
    primary_beam = np.zeros((numPix,numPix))
    for i in range(numPix):
        for j in range(numPix):
            primary_beam[i,j] = np.exp(-1e-4*((i-78)**2+(j-56)**2))
    primary_beam /= np.max(primary_beam)
    
    kwargs_data_no_pb = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg, inverse=True)
    data_class_no_pb = ImageData(**kwargs_data_no_pb)
    
    kwargs_data_with_pb = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg, inverse=True)
    kwargs_data_with_pb['antenna_primary_beam'] = primary_beam
    data_class_with_pb = ImageData(**kwargs_data_with_pb)
    
    kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': np.ones((1,1))}
    psf_class = PSF(**kwargs_psf)
    
    kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}
    kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.04}
    lens_model_list = ['SPEP', 'SHEAR']
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)
            
    kwargs_sersic = {'amp': 30., 'R_sersic': 0.3, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
    lens_light_model_list = ['SERSIC']
    kwargs_lens_light = [kwargs_sersic]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
    
    kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                     'e1': 0.05, 'e2': 0.02}
    source_model_list = ['SERSIC_ELLIPSE']
    kwargs_source = [kwargs_sersic_ellipse]
    source_model_class = LightModel(light_model_list=source_model_list)
    
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
    
    imageModel_no_pb = ImageModel(data_class_no_pb, psf_class, lens_model_class, source_model_class, lens_light_model_class, kwargs_numerics=kwargs_numerics)
    image_sim_no_pb = imageModel_no_pb.image(kwargs_lens, kwargs_source,kwargs_lens_light)
    
    imageModel_with_pb = ImageModel(data_class_with_pb, psf_class, lens_model_class, source_model_class, lens_light_model_class, kwargs_numerics=kwargs_numerics)
    image_sim_with_pb = imageModel_with_pb.image(kwargs_lens, kwargs_source,kwargs_lens_light)
    
    npt.assert_almost_equal(image_sim_with_pb, image_sim_no_pb * primary_beam, decimal=8)