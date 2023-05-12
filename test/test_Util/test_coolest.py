import pytest
import numpy.testing as npt
import numpy as np
import unittest
import os

from lenstronomy.Util.coolest_interface import create_lenstronomy_from_coolest,update_coolest_from_lenstronomy,create_kwargs_mcmc_from_chain_list

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Plots.model_plot import ModelPlot

from lenstronomy.Workflow.fitting_sequence import FittingSequence

class TestCOOLESTinterface(object):

    def test_load(self):
        path = os.getcwd()
        if path[-23:] == 'lenstronomy/lenstronomy':
            path+='/test/test_Util'
        kwargs_out = create_lenstronomy_from_coolest(path+"/coolest_template")
        print(kwargs_out)
        return
    def test_update(self):
        path = os.getcwd()
        if path[-23:] == 'lenstronomy/lenstronomy':
            path+='/test/test_Util'
        kwargs_result={"kwargs_lens":[{'gamma1':0.1,'gamma2':-0.05,'ra_0':0.,'dec_0':0.},
                                      {'theta_E': 0.7, 'e1': -0.15, 'e2': 0.01,
                                       'center_x': 0.03, 'center_y': 0.01}],
                       "kwargs_source":[{'amp':15.,'R_sersic':0.11,'n_sersic':3.6,'center_x':0.02,
                                         'center_y':-0.03,'e1':0.1,'e2':-0.2},
                                        {'amp': np.array([70., 33., 2.1, 3.9, 15., -16., 2.8, -1.7, -4.1, 0.2]),
                                         'n_max': 3, 'beta': 0.1, 'center_x': 0.1, 'center_y': 0.0}],
                       "kwargs_lens_light":[{'amp':11.,'R_sersic':0.2,'n_sersic':3.,'center_x':0.03,
                                             'center_y':0.01,'e1':-0.15,'e2':0.01},
                                            {'amp':12.,'R_sersic':0.02,'n_sersic':6.,'center_x':0.03,
                                             'center_y':0.01,'e1':0.,'e2':-0.15}]}
        update_coolest_from_lenstronomy(path+"/coolest_template",kwargs_result,ending="_update")
        kwargs_out = create_lenstronomy_from_coolest(path+"/coolest_template_update")
        npt.assert_almost_equal(kwargs_out['kwargs_params']['lens_model'][0][1]['e1'],
                                kwargs_result['kwargs_lens'][1]['e1'], decimal=4)
        npt.assert_almost_equal(kwargs_out['kwargs_params']['lens_model'][0][1]['e2'],
                                kwargs_result['kwargs_lens'][1]['e2'], decimal=4)

        return

    def test_full(self):
        # use read json ; create an image ; create noise ; do fit (PSO for result + MCMC for chain)
        # create the kwargs mcmc ; upadte json
        path = os.getcwd()
        if path[-23:] == 'lenstronomy/lenstronomy':
            path+='/test/test_Util'

        kwargs_out = create_lenstronomy_from_coolest(path+"/coolest_template")

        # IMAGE specifics
        background_rms = .005  # background noise per pixel
        exp_time = 500.  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        # PSF : easier for test to create a gaussian PSF
        fwhm = 0.05  # full width at half maximum of PSF
        psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'

        # lensing quantities to create an image
        lens_model_list = kwargs_out['kwargs_model']['lens_model_list']
        kwargs_sie = {'theta_E': .66, 'center_x': 0.05, 'center_y': 0, 'e1': -0.1,
                        'e2': 0.1}  # parameters of the deflector lens model
        kwargs_shear = {'gamma1': 0.0, 'gamma2': -0.05}  # shear values to the source plane
        kwargs_lens = [kwargs_shear, kwargs_sie]
        lens_model_class = LensModel(lens_model_list)

        # Sersic parameters in the initial simulation for the source
        source_model_list = kwargs_out['kwargs_model']['source_light_model_list']
        kwargs_sersic = {'amp': 16, 'R_sersic': 0.1, 'n_sersic': 3.5, 'e1': -0.1, 'e2': 0.1,
                         'center_x': 0.1, 'center_y': 0}
        kwargs_shapelets = {'amp': np.array([ 70.,  33.,   2.1,   3.9 ,  15., -16.,   2.8,  -1.7, -4.1, 0.2]),
                            'n_max': 3, 'beta': 0.1, 'center_x': 0.1, 'center_y': 0.0}
        kwargs_source = [kwargs_sersic, kwargs_shapelets]
        source_model_class = LightModel(source_model_list)

        # Sersic parameters in the initial simulation for the lens light
        lens_light_model_list = kwargs_out['kwargs_model']['lens_light_model_list']
        kwargs_sersic_lens1 = {'amp': 16, 'R_sersic': 0.6, 'n_sersic': 2.5, 'e1': -0.1, 'e2': 0.1, 'center_x': 0.05,
                              'center_y': 0}
        kwargs_sersic_lens2 = {'amp': 3, 'R_sersic': 0.7, 'n_sersic': 3, 'e1': -0.1, 'e2': 0.1, 'center_x': 0.05,
                               'center_y': 0}
        kwargs_lens_light = [kwargs_sersic_lens1,kwargs_sersic_lens2]
        lens_light_model_class = LightModel(lens_light_model_list)

        numPix = 100
        kwargs_out['kwargs_data']['background_rms']=background_rms
        kwargs_out['kwargs_data']['exposure_time'] = exp_time
        kwargs_out['kwargs_data']['image_data'] = np.zeros((numPix,numPix))
        kwargs_out['kwargs_data'].pop('noise_map')

        data_class = ImageData(**kwargs_out['kwargs_data'])
        #PSF
        pixel_scale = kwargs_out['kwargs_data']['transform_pix2angle'][1][1] / kwargs_out['kwargs_psf']['point_source_supersampling_factor']
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': pixel_scale, 'truncation': 3}
        kwargs_out['kwargs_psf'] = kwargs_psf
        psf_class = PSF(**kwargs_out['kwargs_psf'])

        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}


        imageModel = ImageModel(data_class, psf_class, lens_model_class=lens_model_class,
                                source_model_class = source_model_class, lens_light_model_class = lens_light_model_class,
                                kwargs_numerics = kwargs_numerics)

        # generate image
        image_model = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light=kwargs_lens_light, kwargs_ps=None)

        poisson = image_util.add_poisson(image_model, exp_time=exp_time)
        bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
        image_real = image_model + poisson + bkg

        data_class.update_data(image_real)
        kwargs_out['kwargs_data']['image_data'] = image_real

        # MODELING
        # Notes :
        # All the lines above were meant to create a mock image
        # The following is basically the only lines of code you will need
        # (after running the "create_lenstronomy_from_coolest" function) when you actually do the
        # modeling on a pre-existing image (with associated noise and psf proveded)
        band_list = [kwargs_out['kwargs_data'], kwargs_out['kwargs_psf'], kwargs_numerics]
        multi_band_list = [band_list]
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'single-band'}
        kwargs_constraints = {'joint_lens_with_light':[[0,1,['center_x','center_y','e1','e2']],
                                                       [1,1,['center_x','center_y','e1','e2']]]}
        kwargs_likelihood = {'check_bounds':True, 'check_positive_flux': True}

        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_out['kwargs_model'],
                                      kwargs_constraints, kwargs_likelihood,
                                      kwargs_out['kwargs_params'])

        n_particules = 200
        n_iterations = 10
        wr = 5
        n_run_mcmc = 10
        n_burn_mcmc = 10
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': n_particules, 'n_iterations': n_iterations}],
                               ['MCMC', {'n_burn': n_burn_mcmc, 'n_run': n_run_mcmc, 'walkerRatio': wr,
                                         'sigma_scale': 0.01}]
                               ]
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()

        modelPlot = ModelPlot(kwargs_data_joint['multi_band_list'], kwargs_out['kwargs_model'], kwargs_result)
        modelPlot._imageModel.image_linear_solve(inv_bool=True, **kwargs_result)
        #the last 2 lines are meant for solving the linear parameters

        # use the function to save mcmc chains in userfriendly mode
        kwargs_mcmc = create_kwargs_mcmc_from_chain_list(chain_list,kwargs_out['kwargs_model'],kwargs_out['kwargs_params'],
                                           kwargs_out['kwargs_data'],kwargs_out['kwargs_psf'],kwargs_numerics,
                                           kwargs_constraints,idx_chain=1)
        # save the results (aka update the COOLEST json)
        update_coolest_from_lenstronomy(path+"/coolest_template",kwargs_result, kwargs_mcmc)

        return
