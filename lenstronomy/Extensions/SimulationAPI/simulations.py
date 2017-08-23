from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.Solver.image_positions import ImagePosition
import astrofunc.util as util
from astrofunc.util import Util_class
from astrofunc.LensingProfiles.gaussian import Gaussian
from lenstronomy.LensAnalysis.lens_analysis import LensAnalysis

import numpy as np
import copy


class Simulation(object):
    """
    simulation class that querries the major class of lenstronomy
    """
    def __init__(self):
        self.gaussian = Gaussian()
        self.util_class = Util_class()

    def data_configure(self, numPix, deltaPix, exposure_time, sigma_bkg):
        """

        :param numPix: number of pixel (numPix x numPix)
        :param deltaPix: pixel size
        :param exposure_time: exposure time
        :param sigma_bkg: background noise (Gaussian sigma)
        :return:
        """
        mean = 0.  # background mean flux (default zero)
        # 1d list of coordinates (x,y) of a numPix x numPix square grid, centered to zero
        x_grid, y_grid, x_0, y_0, ra_0, dec_0, Matrix, Matrix_inv = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, subgrid_res=1)
        # mask (1= model this pixel, 0= leave blanck)
        mask = np.ones_like(x_grid)  # default is model all pixels
        exposure_map = np.ones_like(x_grid) * exposure_time  # individual exposure time/weight per pixel

        kwargs_data = {
            'sigma_background': sigma_bkg, 'mean_background': mean
            , 'deltaPix': deltaPix, 'numPix_xy': (numPix, numPix)
            , 'exp_time': exposure_time, 'exposure_map': exposure_map
            , 'x_coords': x_grid, 'y_coords': y_grid
            , 'zero_point_x': ra_0, 'zero_point_y': dec_0, 'transform_angle2pix': Matrix_inv
            , 'zero_point_ra': x_0, 'zero_point_dec': y_0, 'transform_pix2angle': Matrix
            , 'mask': mask
            , 'image_data': np.zeros_like(x_grid)
            }
        return kwargs_data

    def psf_configure(self, psf_type="gaussian", fwhm=1, kernelsize=11, deltaPix=1, truncate=3, kernel=None):
        """

        :param psf_type:
        :param fwhm:
        :param pixel_grid:
        :return:
        """
        # psf_type: 'NONE', 'gaussian', 'pixel'
        # 'pixel': kernel, kernel_large
        # 'gaussian': 'sigma', 'truncate'
        if psf_type == 'gaussian':
            sigma = util.fwhm2sigma(fwhm)
            sigma_axis = sigma/np.sqrt(2)
            x_grid, y_grid = util.make_grid(kernelsize, deltaPix)
            kernel_large = self.gaussian.function(x_grid, y_grid, amp=1., sigma_x=sigma_axis, sigma_y=sigma_axis, center_x=0, center_y=0)
            kernel_large = util.array2image(kernel_large)
            kwargs_psf = {'psf_type': psf_type, 'sigma': sigma, 'truncate': truncate*sigma, 'kernel_large': kernel_large}
        elif psf_type == 'pixel':
            kernel_large = copy.deepcopy(kernel)
            kernel_large = self.util_class.cut_psf(kernel_large, psf_size=kernelsize)
            kernel_small = copy.deepcopy(kernel)
            kernel_small = self.util_class.cut_psf(kernel_small, psf_size=kernelsize)
            kwargs_psf = {'psf_type': "pixel", 'kernel': kernel_small, 'kernel_large': kernel_large}
        elif psf_type == 'NONE':
            kwargs_psf = {}
        else:
            raise ValueError("psf type %s not supported!" % psf_type)
        return kwargs_psf

    def im_sim(self, kwargs_options, kwargs_data, kwargs_psf, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else, no_noise=False):
        """

        :param kwargs_options:
        :param kwargs_data:
        :param kwargs_psf:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_else:
        :return:
        """
        lensModel = LensModel(kwargs_options)
        imPos = ImagePosition(lensModel=lensModel)
        if kwargs_options.get('point_source', False):
            deltaPix = kwargs_data['deltaPix']/10.
            numPix = kwargs_data['numPix_xy'][0]*10
            sourcePos_x = kwargs_else['sourcePos_x']
            sourcePos_y = kwargs_else['sourcePos_y']
            x_mins, y_mins = imPos.image_position(sourcePos_x, sourcePos_y, deltaPix, numPix, kwargs_lens, kwargs_else)
            n = len(x_mins)
            mag_list = np.zeros(n)
            for i in range(n):
                potential, alpha1, alpha2, kappa, gamma1, gamma2, mag = lensModel.all(x_mins[i], y_mins[i], kwargs_lens, kwargs_else)
                mag_list[i] = abs(mag)
            kwargs_else['ra_pos'] = x_mins
            kwargs_else['dec_pos'] = y_mins
            kwargs_else['point_amp'] = mag_list*kwargs_else['quasar_amp']

        # update kwargs_else
        makeImage = MakeImage(kwargs_options=kwargs_options, kwargs_data=kwargs_data, kwargs_psf=kwargs_psf)
        image, error_map = makeImage.image_with_params(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_else)
        image = makeImage.Data.array2image(image)
        # add noise
        if no_noise:
            return image
        else:
            poisson = util.add_poisson(image, exp_time=util.array2image(kwargs_data['exposure_map']))
            bkg = util.add_background(image, sigma_bkd=kwargs_data['sigma_background'])
            return image + bkg + poisson

    def fermat_potential(self, kwargs_options, kwargs_data, kwargs_lens, kwargs_else, no_noise=False):
        """
        computes the Fermat potential
        :param kwargs_options:
        :param kwargs_data:
        :param kwargs_lens:
        :param kwargs_else:
        :param no_noise:
        :return: array of Fermat potential for all image positions (in ordering of kwargs_else['ra_pos'])
        """
        lensAnalysis = LensAnalysis(kwargs_options, kwargs_data)
        fermat_pot = lensAnalysis.fermat_potential(kwargs_lens, kwargs_else)
        return fermat_pot