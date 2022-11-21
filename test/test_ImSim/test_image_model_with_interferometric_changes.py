import numpy.testing as npt
import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import scipy.signal

"""
test the antenna primary beam and interferometric PSF (containing negative pixels and do not want normalisation)
The idea of this test is to define two sets of data class and psf class, one with the antenna primary beam and PSF, one without, 
and compare the (image_with_pb_and_psf) with scipy.signal.fftconvolve(image_without_pb_psf * pb, PSF, mode='same').
"""

def test_interferometric_changes():

    sigma_bkg = .05  # background noise per pixel
    exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    numPix = 100  # cutout pixel size
    deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)

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
    
    # define two data classes
    kwargs_data_no_pb = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
    data_class_no_pb = ImageData(**kwargs_data_no_pb)

    kwargs_data_with_pb = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
    kwargs_data_with_pb['antenna_primary_beam'] = primary_beam
    data_class_with_pb = ImageData(**kwargs_data_with_pb)

    # define two psf classes
    kwargs_psf_none = {'psf_type': 'NONE'}
    psf_class_none = PSF(**kwargs_psf_none)

    kernel_cut = kernel_util.cut_psf(psf_test, 201, normalisation = False)
    kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': deltaPix, 'kernel_point_source': kernel_cut,'kernel_point_source_normalisation': False }
    psf_class = PSF(**kwargs_psf)

    # define lens model and source model
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

    # make images using 1) data and psf classes without pb and or psf
    imageModel_no_pb_psf = ImageModel(data_class_no_pb, psf_class_none, lens_model_class, source_model_class, lens_light_model_class, kwargs_numerics=kwargs_numerics)
    image_sim_no_pb_psf = imageModel_no_pb_psf.image(kwargs_lens, kwargs_source,kwargs_lens_light)

    # make images using 2) data and psf classes with defined pb and or psf
    imageModel_with_pb_psf = ImageModel(data_class_with_pb, psf_class, lens_model_class, source_model_class, lens_light_model_class, kwargs_numerics=kwargs_numerics)
    image_sim_with_pb_psf = imageModel_with_pb_psf.image(kwargs_lens, kwargs_source,kwargs_lens_light)

    # add pb and psf to 1) out of the imageModel, compare them to check if the pb and psf changes make sense
    image_sim_with_pb_psf2 = scipy.signal.fftconvolve(image_sim_no_pb_psf * primary_beam,kernel_cut,mode='same')
    npt.assert_almost_equal(image_sim_with_pb_psf, image_sim_with_pb_psf2, decimal=8)