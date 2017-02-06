__author__ = 'sibirrer'

import numpy as np
import astropy.io.fits as pyfits
import scipy.ndimage.filters as filters

import astrofunc.util as util
from astrofunc.DataAnalysis.analysis import Analysis
from lenstronomy.FunctionSet.shapelets import Shapelets


class DataSetup(object):
    """
    this class is meant to set up the information from a lens system class and to make a keyword argument list of all
    needed properties to deal in the MCMC and PSO process
    :param object: takes e.g. StrongLensSystem system
    """
    #TODO has to be replaced/became part of astroObject project
    def __init__(self, system):
        self.system = system
        self.analysis = Analysis()
        self.shapelets = Shapelets()

    def get_data_kwargs(self, image_name, numPix):
        """

        :param fits_name: name of fits file with data
        :param numPix: number of pixel rows/columns to be cut out
        :return: keyword argument list which can be processed by the MCMC_sampler
        """
        data = self.system.get_cutout_image(image_name, numPix)
        deltaPix, _ = self.system.get_pixel_scale(image_name)
        deltaPix *= 3600  # convert into arcsec
        ra_pos, dec_pos = self.system.get_image_position()
        mean, sigma_b = self.system.get_background(image_name)
        data_reduced = data - mean
        reduce_frac = self.system.get_exposure_time(image_name)
        try:
            exp_map = self.system.get_exposure_map(image_name)
        except:
            print('No exposure map found in image')
            exp_map = None
        kwargs_data = {'image_data': data_reduced, 'sigma_background': sigma_b, 'mean_background': mean, 'deltaPix': deltaPix, 'reduced_noise': reduce_frac, 'imagePos_ra': ra_pos, 'imagePos_dec': dec_pos, 'exposure_map': exp_map}
        return kwargs_data

    def get_data_kwargs_new(self, image_name, numPix, sigma_b=None, mean=None):
        """
        same as get_data_kwargs but without calling sextractor
        :param image_name:
        :param numPix:
        :param sigma_b:
        :param mean:
        :return:
        """
        data = self.system.get_cutout_image(image_name, numPix)
        deltaPix, _ = self.system.get_pixel_scale(image_name)
        deltaPix *= 3600  # convert into arcsec
        ra_pos, dec_pos = self.system.get_image_position()
        if sigma_b is None or mean is None:
            mean, sigma_b = self.system.get_background(image_name)
        data_reduced = data - mean
        reduce_frac = self.system.get_exposure_time(image_name)
        x_0, y_0 = self.system.get_pixel_zero_point(image_name)
        Matrix = self.system.get_transform_matrix_angle2pix(image_name)
        cos_dec = np.cos(self.system.dec/360*2*np.pi)
        #try:
        exp_map = self.system.get_exposure_map(image_name)
        exp_map[exp_map <= 0] = 10**(-10)
        #except:
        #    print('No exposure map found in image')
        #    exp_map = None
        ra_coords, dec_coords = self.system.get_coordinate_grid(image_name)
        x_coords, y_coords = self.system.get_coordinate_grid_relative(image_name)
        kwargs_data = {'image_data': data_reduced, 'sigma_background': sigma_b, 'mean_background': mean
            ,'deltaPix': deltaPix, 'reduced_noise': reduce_frac, 'imagePos_x': ra_pos, 'imagePos_y': dec_pos
            ,'exposure_map': exp_map, 'ra_coords': ra_coords, 'dec_coords': dec_coords, 'x_coords': x_coords, 'y_coords': y_coords
            , 'zero_point_x': x_0, 'zero_point_y': y_0, 'transform_angle2pix': Matrix}
        return kwargs_data

    def get_psf_kwargs(self, image_name, psf_type, psf_size = None):
        """
        returns the keyword arguments for the psf configuration fitted on the data image
        :param image_name:
        :param psf_type:
        :return:
        """
        if psf_type == 'gaussian':
            mean_list = self.system.get_psf_fit(image_name, psf_type)
            sigma = mean_list[1]*self.system.get_pixel_scale(image_name)
            return {'psf_type': psf_type, 'sigma': sigma}
        if psf_type == 'moffat':
            mean_list = self.system.get_psf_fit(image_name, psf_type)
            alpha = mean_list[1]
            beta = mean_list[2]
            alpha *= self.system.get_pixel_scale(image_name)
            return {'psf_type': psf_type, 'alpha': alpha, 'beta': beta}
        if psf_type == 'pixel':
            kernel= self.system.get_psf_kernel(image_name)
            kernel = util.cut_edges(kernel, psf_size)
            return {'psf_type': psf_type, 'kernel': kernel}
        raise ValueError('psf type %s not in list' %psf_type)

    def get_psf_kwargs_update(self, image_name, psf_type, psf_size=None, psf_size_large=91, filter_object=None, kwargs_cut={}):
        """
        does the same as get_psf_kwargs but can also restrict itself to specially chosen objects
        :param image_name:
        :param psf_type:
        :param psf_size:
        :param filter_object:
        :return:
        """
        exp_time = self.system.get_exposure_time(image_name)
        HDUFile, image_no_border = self.system.get_HDUFile(image_name)

        kernel_large, mean_list, restrict_psf, star_list = self.analysis.get_psf_outside(HDUFile, image_no_border, exp_time, psf_type, filter_object, kwargs_cut)
        if psf_type == 'gaussian':
            sigma = mean_list[1]*self.system.get_pixel_scale(image_name)
            psf_kwargs = {'psf_type': psf_type, 'sigma': sigma}
        elif psf_type == 'moffat':
            alpha = mean_list[1]
            beta = mean_list[2]
            alpha *= self.system.get_pixel_scale(image_name)
            psf_kwargs = {'psf_type': psf_type, 'alpha': alpha, 'beta': beta}
        elif psf_type == 'pixel':
            kernel = util.cut_edges(kernel_large, psf_size)
            kernel = util.kernel_norm(kernel)
            kernel_large = util.cut_edges(kernel_large, psf_size_large)
            kernel_large = util.kernel_norm(kernel_large)
            kernel_list = []
            for i in range(len(star_list)):
                if i == 0:
                   kernel_list.append(kernel_large)
                else:
                    star = star_list[i]
                    kernel_star = util.cut_edges(star, psf_size_large)
                    kernel_star = util.kernel_norm(kernel_star)
                    kernel_list.append(kernel_star-kernel_large)
            psf_kwargs = {'psf_type': psf_type, 'kernel': kernel, 'kernel_large': kernel_large, 'kernel_list': kernel_list}
        else:
            raise ValueError('psf type %s not in list' % psf_type)
        return psf_kwargs, restrict_psf, star_list

    def get_psf_from_fits(self, path2fits, psf_type, psf_size, psf_size_large=91):
        """
        ment to import a psf from Tiny Tim
        :param path2fits: path to the fits file
        :return:
        """
        psf_data = pyfits.getdata(path2fits)
        kernel = util.cut_edges_TT(psf_data, psf_size)
        kernel_large = util.cut_edges_TT(psf_data, psf_size_large)
        kernel_large = util.kernel_norm(kernel_large)
        kernel_large_norm = np.copy(kernel_large)
        kernel = util.kernel_norm(kernel)
        psf_kwargs = {'psf_type': psf_type, 'kernel': kernel, 'kernel_large':kernel_large_norm}
        return psf_kwargs

    def get_psf_from_system(self, image_name, psf_size, psf_size_large=91):
        """
        ment to import a psf from Tiny Tim
        :param path2fits: path to the fits file
        :return:
        """
        psf_data = self.system.get_psf_data(image_name)
        psf_kwargs = self.cut_psf(psf_data, psf_size, psf_size_large)
        return psf_kwargs

    def cut_psf(self, psf_data, psf_size, psf_size_large):
        """
        cut the psf properly
        :param psf_data:
        :param psf_size:
        :return:
        """
        kernel = util.cut_edges_TT(psf_data, psf_size)
        kernel_large = util.cut_edges_TT(psf_data, psf_size_large)
        kernel_large = util.kernel_norm(kernel_large)
        kernel_large_norm = np.copy(kernel_large)
        kernel = util.kernel_norm(kernel)
        psf_kwargs = {'psf_type': 'pixel', 'kernel': kernel, 'kernel_large': kernel_large_norm}
        return psf_kwargs

    def get_psf_errors(self, psf_kwargs, data_kwargs, star_list):
        """
        returns a error map of sigma prop Intensity for a stacked psf estimation
        :param psf_kwargs:
        :param star_list:
        :return:
        """
        psf_size = len(psf_kwargs['kernel_large'])
        kernel_mean = util.image2array(psf_kwargs['kernel_large'])
        weights = np.zeros(len(star_list))
        cov_i = np.zeros((psf_size**2,psf_size**2))
        num_stars = len(star_list)
        for i in range(0,num_stars):
            star_list_i = star_list[i].copy()
            star = util.cut_edges(star_list_i, psf_size)
            weights[i] = np.sum(star)
            rel_array = np.array([util.image2array(star)/weights[i]-kernel_mean])
            a = (rel_array.T).dot(rel_array)
            cov_i += a
        factor = 1./(num_stars -1)
        #weights_sum = sum(weights)
        sigma2_stack = factor*util.array2image(np.diag(cov_i))
        psf_stack = psf_kwargs['kernel_large'].copy()
        sigma2_stack_new = sigma2_stack# - (data_kwargs['sigma_background']**2/weights_sum)
        sigma2_stack_new[np.where(sigma2_stack_new < 0)] = 0
        psf_stack[np.where(psf_stack < data_kwargs['sigma_background'])] = data_kwargs['sigma_background']
        error_map = sigma2_stack_new/(psf_stack)**2
        #error_map[np.where(error_map < psf_stack**2/data_kwargs['reduced_noise'])] = 0
        # n = len(error_map)
        #error_map[(n-1)/2-1:(n-1)/2+2,(n-1)/2-1:(n-1)/2+2] += 0
        error_map = filters.gaussian_filter(error_map, sigma=0.5)
        return error_map

    def get_lens_light_profile(self, image_name, numPix, mask, psf_kwargs, n_particles=240, n_iterations=200, threadCount=6, lens_light_type='TRIPLE_SERSIC', lowerLimit=None, upperLimit=None):
        """

        :param system: system class
        :param image_name: name of image
        :param mask: mask of image
        :return: kwargs for light profile
        """
        from lenstronomy.DataAnalysis.psf_fitting import Fitting
        fitting = Fitting()
        image = self.system.get_cutout_image(image_name, numPix)
        mean, sigma_b = self.system.get_background(image_name)
        ra_coords, dec_coords = self.system.get_coordinate_grid_relative(image_name)
        reduce_frac = self.system.get_exposure_time(image_name)
        deltaPix, _ = self.system.get_pixel_scale(image_name)
        deltaPix *= 3600  # convert into arcsec
        if lens_light_type == 'TRIPLE_SERSIC':
            kwargs_lens_light, model, particles = fitting.sersic_triple_fit(image, ra_coords, dec_coords, sigma_b, reduce_frac, mask, deltaPix, psf_kwargs, n_particles, n_iterations, lowerLimit, upperLimit, threadCount)
        elif lens_light_type == 'SERSIC':
            kwargs_lens_light, model, particles = fitting.sersic_fit(image, ra_coords, dec_coords, sigma_b, reduce_frac, mask, deltaPix, psf_kwargs, n_particles, n_iterations, lowerLimit, upperLimit, threadCount)
        elif lens_light_type == 'SERSIC_2':
            kwargs_lens_light, model, particles = fitting.sersic2_fit(image, ra_coords, dec_coords,  sigma_b, reduce_frac, mask, deltaPix, psf_kwargs, n_particles, n_iterations, lowerLimit, upperLimit, threadCount)
        elif lens_light_type == 'CORE_SERSIC':
            kwargs_lens_light, model, particles = fitting.core_sersic_fit(image, ra_coords, dec_coords, sigma_b, reduce_frac, mask, deltaPix, psf_kwargs, n_particles, n_iterations, lowerLimit, upperLimit, threadCount)
        else:
            raise ValueError("no lens_light_type named %s available" % lens_light_type)
        return kwargs_lens_light, model, particles