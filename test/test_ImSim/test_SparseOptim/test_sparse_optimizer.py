import numpy as np
import numpy.testing as npt
import pytest
import unittest
import matplotlib.pyplot as plt

from lenstronomy.ImSim.SparseOptim.sparse_optimizer import SparseOptimizer
from lenstronomy.Util import util


import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf

import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.util as lenstro_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

from lenstronomy.ImSim.image_sparse_solve import ImageSparseFit

from MuSLIT.utils.plot import nice_colorbar

%matplotlib inline


class TestSparseOptimizer(object):
    """
    class to test SparseOptimizer
    """
    def setup(self):
        # data specifics
        background_rms = 5 # background noise per pixel
        psf_fwhm = 0.4  # full width half max of PSF, in delta_pix units

        num_pix = 49  # cutout pixel size
        delta_pix = 0.2  # pixel size in arcsec (area per pixel = deltaPix**2) -->  if 1, means you we work in pixel units

        # data specification (coordinates, etc.)
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = lenstro_util.make_grid_with_coordtransform(numPix=num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)

        kwargs_data = {
            'background_rms': background_rms,
            #'exposure_time': np.ones((num_pix, num_pix)) * exp_time,  # individual exposure time/weight per pixel
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((num_pix, num_pix))
        }
        data_class = ImageData(**kwargs_data)

        # PSF specification
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm, 'pixel_size': delta_pix, 'truncation': 11}
        #kwargs_psf = {'psf_type': 'NONE'}
        psf_class = PSF(**kwargs_psf)

        lens_model_list = ['SPEMD']
        kwargs_spemd = {'theta_E': 2, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}
        kwargs_lens = [kwargs_spemd]#, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # list of source light profiles
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_sersic_ellipse_source = {'amp': 2000, 'R_sersic': 0.6, 'n_sersic': 1, 'e1': 0, 'e2': 0,
                                        'center_x': 0.6, 'center_y': 0.6}
        kwargs_source = [kwargs_sersic_ellipse_source]
        source_model_class = LightModel(light_model_list=source_model_list)

        # list of lens light profiles
        lens_light_model_list = []
        kwargs_lens_light = [{}]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

        # get the simalated lens image (i.e. image plane)
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, 
                                lens_light_model_class, point_source_class=None, 
                                kwargs_numerics=kwargs_numerics)

        image_sim_no_noise = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps=None)

        bkg_noise = image_util.add_background(image_sim_no_noise, sigma_bkd=background_rms)
        image_sim = image_sim_no_noise + bkg_noise

        kwargs_data['image_data'] = image_sim
        data_class.update_data(image_sim)


        # create the optimizer
        self.sparseOptimizer = SparseOptimizer(image_data, psf_kernel, sigma_bkg, 
                                               source_profile, lens_light_profile=None)
