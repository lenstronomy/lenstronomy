__author__ = 'sibirrer'

import numpy as np
from astrofunc.util import Util_class
from lenstronomy.Cosmo.time_delay_sampling import TimeDelaySampling
from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.MCMC.compare import Compare
from lenstronomy.Workflow.parameters import Param


class MCMC_chain(object):
    """
    this class contains the routines to run a MCMC process with one single image
    """
    def __init__(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else):
        """
        initializes all the classes needed for the chain
        """
        # print('initialized on cpu', threading.current_thread())
        self.util_class = Util_class()
        self.subgrid_res = kwargs_options['subgrid_res']
        self.data = kwargs_data['image_data']
        self.numPix = len(self.data)
        self.sigma_b = kwargs_data['sigma_background']
        self.deltaPix = kwargs_data['deltaPix'] #pixel size in arc seconds
        self.num_shapelets = kwargs_options.get('shapelet_order', -1)
        exposure_map = kwargs_data.get('exposure_map', None)
        if exposure_map is None:
            self.exposure_map = kwargs_data.get('reduced_noise', 1)
        else:
            self.exposure_map = exposure_map
        self.mask = kwargs_data.get('mask', 1)
        self.mask_lens_light = kwargs_data.get('mask_lens_light', 1)
        self.sampling_option = kwargs_options.get('X2_type', 'image')
        self.makeImage = MakeImage(kwargs_options, kwargs_data, kwargs_psf)
        self.param = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else)
        self.compare = Compare(kwargs_options)
        self.lowerLimit, self.upperLimit = self.param.param_bounds()
        self.timeDelay = TimeDelaySampling()
        self.time_delay = kwargs_options.get('time_delay', False)
        if self.time_delay is True:
            self.delays_measured = kwargs_data['time_delays']
            self.delays_errors = kwargs_data['time_delays_errors']
        self.inv_bool = kwargs_options.get('source_marg', False)  # whether to fully invert the covariance matrix for marginalization
        self.priors_bool = kwargs_options.get('priors', False)
        if self.priors_bool:
            self.kwargs_priors = kwargs_options['kwargs_priors']

    def X2_chain_image(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        im_sim, model_error, cov_matrix, param = self.makeImage.make_image_ideal(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, self.deltaPix, self.subgrid_res, inv_bool=self.inv_bool)
        #compute X^2
        X = self.makeImage.reduced_residuals(im_sim, model_error)
        logL = self.compare.get_log_likelihood(X, cov_matrix=cov_matrix)
        logL -= self.check_bounds(args, self.lowerLimit, self.upperLimit)
        # logL -= self.bounds_convergence(kwargs_lens)
        if self.time_delay is True:
            logL += self.logL_delay(kwargs_lens, kwargs_source, kwargs_else)
        if self.priors_bool:
            logL += self.priors(kwargs_lens, self.kwargs_priors)
        return logL, None

    def X2_chain_lens_light(self, args):
        """
        routine to compute X^2 value of lens light profile
        :param args:
        :return:
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        lens_light, _, _ = self.makeImage.make_image_lens_light(kwargs_lens_light, self.deltaPix, self.subgrid_res)
        #compute X^2
        X = self.makeImage.reduced_residuals(lens_light, lens_light_mask=True)
        logL = self.compare.get_log_likelihood(X)
        logL -= self.check_bounds(args, self.lowerLimit, self.upperLimit)
        if self.priors_bool:
            logL += self.priors(kwargs_lens, self.kwargs_priors)
        return logL, None

    def X2_chain_catalogue(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chain
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        x_mapped, y_mapped = self.makeImage.mapping_IS(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_else, **kwargs_lens)
        #compute X^2
        X2 = self.compare.compare_distance(x_mapped, y_mapped)*1000
        X2 += self.check_bounds(args, self.lowerLimit, self.upperLimit)
        if self.priors_bool:
            X2 -= self.priors(kwargs_lens, self.kwargs_priors)
        return -X2, None

    def X2_chain_combined(self, args):
        X2_cat, _ = self.X2_chain_catalogue(args)
        X2_im,_ = self.X2_chain_image(args)
        return X2_cat/(self.deltaPix/100)**2 + X2_im, None

    def priors(self, kwargs_lens, kwargs_priors):
        """

        :param kwargs_lens:
        :param kwargs_priors:
        :return:
        """
        prior = 0
        if 'gamma_ext' in kwargs_lens and 'gamma_ext' in kwargs_priors and 'gamma_ext_sigma' in kwargs_priors:
            prior -= (kwargs_lens['gamma_ext']-kwargs_priors['gamma_ext'])**2/(2*kwargs_priors['gamma_ext_sigma'])**2
        if 'psi_ext' in kwargs_lens and 'psi_ext' in kwargs_priors and 'psi_ext_sigma' in kwargs_priors:
            prior -= (kwargs_lens['psi_ext']-kwargs_priors['psi_ext'])**2/(2*kwargs_priors['psi_ext_sigma'])**2
        return prior

    def get_source_position(self, kwargs_lens, kwargs_else):
        """
        return source position given lens model and catalogue image positions
        """
        x_mapped, y_mapped = self.makeImage.mapping_IS(kwargs_else['ra_pos'], kwargs_else['dec_pos'], **kwargs_lens)
        return np.mean(x_mapped), np.mean(y_mapped)

    def check_bounds(self, args, lowerLimit, upperLimit):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        penalty = 0
        for i in range(0, len(args)):
            if args[i] < lowerLimit[i] or args[i] > upperLimit[i]:
                penalty = 10**15#np.NaN #10**10
                #print(i, args[i], lowerLimit[i], upperLimit[i])
                #print("warning!!!")
        return penalty

    def bounds_convergence(self, kwargs_lens, kwargs_else=None):
        """
        bounds computed from kwargs
        """
        convergence = self.makeImage.LensModel.kappa(self.x_grid, self.y_grid, **kwargs_lens)
        if np.min(np.array(convergence)) < -0.1:
            return 10**10
        else:
            return 0

    def logL_delay(self, kwargs_lens, kwargs_source, kwargs_else):
        """
        routine to compute the log likelihood of the time delay distance
        :param args:
        :return:
        """
        delay_arcsec = self.makeImage.get_time_delay(kwargs_lens, kwargs_source, kwargs_else)
        D_dt_model = kwargs_else['delay_dist']
        delay_days = self.timeDelay.days_D_model(delay_arcsec, D_dt_model)
        logL = self.compare.delays(delay_days, self.delays_measured, self.delays_errors)
        return logL

    def __call__(self, a):
        if self.sampling_option == 'image':
            return self.X2_chain_image(a)
        elif self.sampling_option == 'catalogue':
            return self.X2_chain_catalogue(a)
        elif self.sampling_option == 'lens_light':
            return self.X2_chain_lens_light(a)
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)

    def likelihood(self, a):
        if self.sampling_option == 'image':
            return self.X2_chain_image(a)
        elif self.sampling_option == 'catalogue':
            return self.X2_chain_catalogue(a)
        elif self.sampling_option == 'lens_light':
            return self.X2_chain_lens_light(a)
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)

    def computeLikelihood(self, ctx):
        if self.sampling_option == 'image':
            likelihood, _ = self.X2_chain_image(ctx.getParams())
        elif self.sampling_option == 'catalogue':
            likelihood, _ = self.X2_chain_catalogue(ctx.getParams())
        elif self.sampling_option == 'lens_light':
            likelihood, _ = self.X2_chain_lens_light(ctx.getParams())
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)
        return likelihood

    def setup(self):
        pass

    def numData_points(self):
        """
        returns the effective number of data points considered in the X2 estimation to compute the reduced X2 value
        """
        if type(self.mask) == int:
            n = self.numPix**2
        else:
            if self.sampling_option == 'lens_light':
                n = np.sum(self.mask_lens_light)
            else:
                n = np.sum(self.mask)
        num_param, _ = self.param.num_param()
        num_shapelets = (self.num_shapelets + 1)*(self.num_shapelets + 2)/2
        return n - num_param - num_shapelets - 1


class MCMC_multiband_chain(object):
    """
    class for computing the likelihood of multiband fitting
    """
    def __init__(self, kwargs_data_list, kwargs_psf_list, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else):
        """
        initializes all the classes needed for the chain
        """
        kwargs_data1 = kwargs_data_list[0]
        kwargs_data2 = kwargs_data_list[1]
        kwargs_psf1 = kwargs_psf_list[0]
        kwargs_psf2 = kwargs_psf_list[1]
        self.util_class = Util_class()
        self.subgrid_res = kwargs_options['subgrid_res']
        self.num_shapelets = kwargs_options.get('shapelet_order', -1)
        self.sampling_option = kwargs_options.get('X2_type', 'image')

        self.param = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else)
        self.compare = Compare(kwargs_options)

        self.lowerLimit, self.upperLimit = self.param.param_bounds()
        self.timeDelay = TimeDelaySampling()
        self.time_delay = kwargs_options.get('time_delay', False)
        if self.time_delay is True:
            self.delays_measured = kwargs_data_list[2]['time_delays']  # ATTENTION: changed from coming from kwargs_data to kwargs_data[3]
            self.delays_errors = kwargs_data_list[2]['time_delays_errors']  # ATTENTION: changed from coming from kwargs_data to kwargs_data[3]
        self.inv_bool = kwargs_options.get('source_marg', False) #  whether to fully invert the covariance matrix for marginalization

        self.data1 = kwargs_data1['image_data']
        self.numPix1 = len(self.data1)
        self.sigma_b1 = kwargs_data1['sigma_background']
        self.deltaPix1 = kwargs_data1['deltaPix'] #pixel size in arc seconds
        exposure_map1 = kwargs_data1.get('exposure_map', None)
        if exposure_map1 is None:
            self.exposure_map1 = kwargs_data1.get('reduced_noise', 1)
        else:
            self.exposure_map1 = exposure_map1
        self.mask1 = kwargs_data1.get('mask', 1)  # ATTENTION: changed from coming from kwargs_option to kwargs_data
        self.mask_lens_light1 = kwargs_data1.get('mask_lens_light', 1)  # ATTENTION: changed from coming from kwargs_option to kwargs_data
        self.makeImage1 = MakeImage(kwargs_options, kwargs_data1, kwargs_psf1)
        self.x_grid1, self.y_grid1 = self.util_class.make_subgrid(kwargs_data1['x_coords'], kwargs_data1['y_coords'], self.subgrid_res)

        self.data2 = kwargs_data2['image_data']
        self.numPix2 = len(self.data2)
        self.sigma_b2 = kwargs_data2['sigma_background']
        self.deltaPix2 = kwargs_data2['deltaPix'] #pixel size in arc seconds
        exposure_map2 = kwargs_data2.get('exposure_map', None)
        if exposure_map2 is None:
            self.exposure_map2 = kwargs_data2.get('reduced_noise', 2)
        else:
            self.exposure_map2 = exposure_map2
        self.mask2 = kwargs_data2.get('mask', 1)  # ATTENTION: changed from coming from kwargs_option to kwargs_data
        self.mask_lens_light2 = kwargs_data2.get('mask_lens_light', 1)  # ATTENTION: changed from coming from kwargs_option to kwargs_data
        self.makeImage2 = MakeImage(kwargs_options, kwargs_data2, kwargs_psf2)
        self.x_grid2, self.y_grid2 = self.util_class.make_subgrid(kwargs_data2['x_coords'], kwargs_data2['y_coords'], self.subgrid_res)

    def X2_chain_image(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chainF
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        im_sim1, model_error1, cov_matrix1, param1 = self.makeImage1.make_image_ideal(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, self.deltaPix1, self.subgrid_res, inv_bool=self.inv_bool)
        im_sim2, model_error2, cov_matrix2, param2 = self.makeImage2.make_image_ideal(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, self.deltaPix2, self.subgrid_res, inv_bool=self.inv_bool)
        #im_sim = util.array2image(im_sim)
        #compute X^2
        logL1 = self.compare.get_log_likelihood(im_sim1, self.data1, self.sigma_b1, self.exposure_map1, mask=self.mask1, model_error=model_error1, cov_matrix=cov_matrix1)
        logL2 = self.compare.get_log_likelihood(im_sim2, self.data2, self.sigma_b2, self.exposure_map2, mask=self.mask2, model_error=model_error2, cov_matrix=cov_matrix2)
        logL = logL1 + logL2
        logL -= self.check_bounds(args, self.lowerLimit, self.upperLimit)
        logL -= self.bounds_convergence(kwargs_lens)
        if self.time_delay is True:
            logL += self.logL_delay(kwargs_lens, kwargs_source, kwargs_else)
        return logL, None

    def X2_chain_catalogue(self, args):
        """
        routine to compute X2 given variable parameters for a MCMC/PSO chain
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        x_mapped, y_mapped = self.makeImage1.mapping_IS(kwargs_else['ra_pos'], kwargs_else['dec_pos'], kwargs_else, **kwargs_lens)
        #compute X^2
        X2 = self.compare.compare_distance(x_mapped, y_mapped)*1000
        X2 += self.check_bounds(args, self.lowerLimit, self.upperLimit)
        return -X2, None

    def X2_chain_lens_light(self, args):
        """
        routine to compute X^2 value of lens light profile
        :param args:
        :return:
        """
        #extract parameters
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else = self.param.get_all_params(args)
        #generate image
        lens_light1 = self.makeImage1.make_image_lens_light(kwargs_lens_light, self.deltaPix1, self.subgrid_res)
        lens_light2 = self.makeImage2.make_image_lens_light(kwargs_lens_light,
                                                             self.deltaPix2, self.subgrid_res)
        #compute X^2
        logL1 = self.compare.get_log_likelihood(lens_light1, self.data1, self.sigma_b1, self.exposure_map1, mask=self.mask_lens_light1)
        logL2 = self.compare.get_log_likelihood(lens_light2, self.data2, self.sigma_b2, self.exposure_map2,
                                                mask=self.mask_lens_light2)
        logL = logL1 + logL2
        logL -= self.check_bounds(args, self.lowerLimit, self.upperLimit)
        return logL, None

    def check_bounds(self, args, lowerLimit, upperLimit):
        """
        checks whether the parameter vector has left its bound, if so, adds a big number
        """
        penalty = 0
        for i in range(0,len(args)):
            if args[i] < lowerLimit[i] or args[i] > upperLimit[i]:
                penalty = 10**15#np.NaN #10**10
        return penalty

    def bounds_convergence(self, kwargs_lens, kwargs_else=None):
        """
        bounds computed from kwargs
        """
        convergence1 = self.makeImage1.LensModel.kappa(self.x_grid1, self.y_grid1, **kwargs_lens)
        convergence2 = self.makeImage2.LensModel.kappa(self.x_grid2, self.y_grid2, **kwargs_lens)
        if np.min(np.array(convergence1)) < -0.1 or np.min(np.array(convergence2)) < -0.1:
            return 10**10
        else:
            return 0

    def logL_delay(self, kwargs_lens, kwargs_source, kwargs_else):
        """
        routine to compute the log likelihood of the time delay distance
        :param args:
        :return:
        """
        delay_arcsec = self.makeImage1.get_time_delay(kwargs_lens, kwargs_source, kwargs_else)
        D_dt_model = kwargs_else['delay_dist']
        delay_days = self.timeDelay.days_D_model(delay_arcsec, D_dt_model)
        logL = self.compare.delays(delay_days, self.delays_measured, self.delays_errors)
        return logL

    def numData_points(self):
        """
        returns the effective number of data points considered in the X2 estimation to compute the reduced X2 value
        """
        if type(self.mask1) == int:
            n1 = self.numPix1**2
        else:
            if self.sampling_option == 'lens_light':
                n1 = np.sum(self.mask_lens_light1)
            else:
                n1 = np.sum(self.mask1)
        if type(self.mask2) == int:
            n2 = self.numPix2**2
        else:
            if self.sampling_option == 'lens_light':
                n2 = np.sum(self.mask_lens_light2)
            else:
                n2 = np.sum(self.mask2)
        num_param, _ = self.param.num_param()
        num_shapelets = (self.num_shapelets + 1)*(self.num_shapelets + 2)/2
        return n1 + n2 - num_param - num_shapelets*2 - 1

    def __call__(self, a):
        if self.sampling_option == 'image':
            return self.X2_chain_image(a)
        elif self.sampling_option == 'catalogue':
            return self.X2_chain_catalogue(a)
        elif self.sampling_option == 'lens_light':
            return self.X2_chain_lens_light(a)
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)

    def likelihood(self, a):
        if self.sampling_option == 'image':
            return self.X2_chain_image(a)
        elif self.sampling_option == 'catalogue':
            return self.X2_chain_catalogue(a)
        elif self.sampling_option == 'lens_light':
            return self.X2_chain_lens_light(a)
        else:
            raise ValueError('option %s not valid!' %self.sampling_option)

    def computeLikelihood(self, ctx):
        if self.sampling_option == 'image':
            likelihood, _ = self.X2_chain_image(ctx.getParams())
        elif self.sampling_option == 'catalogue':
            likelihood, _ = self.X2_chain_catalogue(ctx.getParams())
        elif self.sampling_option == 'lens_light':
            return self.X2_chain_lens_light(ctx.getParams())
        else:
            raise ValueError('option %s not valid!' % self.sampling_option)

    def setup(self):
        pass