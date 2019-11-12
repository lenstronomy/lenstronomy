# class that implements SLIT algorithm

import numpy as np
import scipy.signal as scp
import matplotlib.pyplot as plt

from lenstronomy.Util import util



class SparseOptimizer(object):


    def __init__(self, image_data, psf_kernel, sigma_bkg, likelihood_mask, 
                 source_profile_class, lens_light_profile_class=None,
                 k_max=5, n_iter=50, S0=[0], scheme='FB', mask=[0], weight_S=1, tau=0, n_weights=1, verbose=False):
        self._image_data = image_data
        # self._noise_map = noise_map
        self._sigma_bkg = sigma_bkg
        self._psf_kernel = psf_kernel
        self._mask = likelihood_mask

        self._source_light = source_profile_class
        self._lens_light   = lens_light_profile_class
        if self._lens_light is not None:
            self._solve_for_lens_light = True
        else:
            self._solve_for_lens_light = False

        (num_pix_x, num_pix_y) = image_data.shape
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x

        self._k_max = k_max
        self._n_iter = n_iter
        self._S0 = S0
        self._scheme = scheme
        self._mask = mask
        self._weight_S = weight_S
        self._tau = tau
        self._n_weights = n_weights

        self._verbose = verbose


    def solve_sparse(self, lensing_operator_class, kwargs_source, kwargs_lens_light=None):
        if self._solve_for_lens_light:
            raise NotImplementedError("SLIT_MCA algorithm not yet implemented")
            # return self._solve_sparse_all(lensing_operator_class, kwargs_source, kwargs_lens_light)
        else:
            return self._solve_sparse_source(lensing_operator_class, kwargs_source)


    def _solve_sparse_source(self, F, kwargs_source):
        """SLIT algorithm"""
        # number of side pixels in image and source planes
        n_img = F.image_plane_num_pix
        n_src = F.source_plane_num_pix

        # noise level in image plane
        noise_img = self._sigma_bkg

        # get noise levels in source plane at starlet scales
        noise_coeffs_src = self.noise_levels_source_plane(F, kwargs_source)



        return np.zeros_like(Y)  # TODO


    @property
    def Y(self):
        if not hasattr(self, 'Y'):
            image_data = self._image_data.copy()
            image_data[self._mask == 1] = noise_img * np.random.randn(image_data.shape)[self._mask == 1]
            self.Y = image_data
        return self.Y


    @property
    def H(self):
        if not hasattr(self, 'H'):
            self.H = self._psf_kernel
        return self.H


    @property
    def HT(self):
        return self.H.T  # TODO : conjugate ?


    def _solve_sparse_all(self, F, kwargs_source, kwargs_lens_light):
        """SLIT_MCA algorithm"""
        pass


    def noise_levels_source_plane(self, lensing_operator_class, kwargs_source):
        if not hasattr(self, '_noise_levels_src'):
            self._noise_levels_src = self._compute_noise_levels_src(lensing_operator_class, kwargs_source)
        return self._noise_levels_src


    def _compute_noise_levels_src(self, F, kwargs_source):
        n_img = F.image_plane_num_pix

        # estimate noise level
        # sigma_bkg = self.sigma_background_mad(self._noise_map)

        # PSF noise map
        HT = self._psf_kernel.T
        HT_noise = np.ones((n_img, n_img)) * self._sigma_bkg * np.sqrt(np.sum(HT**2))
        FT_HT_noise = F.image2source_2d(HT_noise)
        FT_HT_noise[FT_HT_noise == 0] = 10 * np.mean(FT_HT_noise)

        # computes noise levels in in source plane in starlet space
        dirac = self.dirac_impulse(n_img)
        dirac_mapped = F.image2source_2d(dirac)

        # model transform of the impulse
        dirac_coeffs = self._source_light.decomposition(dirac_mapped, **kwargs_source)
        print(dirac_coeffs.shape)

        noise_levels = np.zeros(dirac_coeffs.shape)
        for scale_idx in range(noise_levels.shape[0]):
            dirac_scale = dirac_coeffs[scale_idx, :, :]
            
            # TODO : separate gaussian / pixel convolution 
            levels = scp.fftconvolve(FT_HT_noise**2, dirac_scale**2, mode='same')
            

            levels[levels == 0] = 0
            noise_levels[scale_idx, :, :] = np.sqrt(np.abs(levels))
        return noise_levels


    @property
    def sigma_background_mad(self):
        noise_map = self.error_response
        mad = np.median(np.abs(noise_map - np.median(noise_map)))
        return 1.48 * mad

    # def _forward_backward(self):


    # def _fista(self):


    @staticmethod
    def dirac_impulse(num_pix):
        """
        returns the 1d array of a Dirac impulse at the center of the image

        :return: 1d numpy array of response, 2d array of additonal errors (e.g. point source uncertainties)
        """
        dirac = np.zeros((num_pix, num_pix), dtype=float)
        dirac[int(num_pix/2), int(num_pix/2)] = 1.
        return dirac

