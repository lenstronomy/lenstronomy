# class that implements SLIT algorithm

import numpy as np
import scipy.signal as scp
import matplotlib.pyplot as plt

from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from lenstronomy.Util import util
from lenstronomy.Plots import plot_util
from lenstronomy.ImSim.SparseOptim import algorithms



class SparseOptimizer(object):


    def __init__(self, data_class, source_profile_class, psf_class=None, lens_light_profile_class=None, likelihood_mask=None, 
                 k_max=5, n_iter=50, weight_S=1, n_weights=1, sparsity_prior_norm=1, force_positivity=True, 
                 convolution_type='fft_static', verbose=False):
        
        self._image_data = data_class.data

        (num_pix_x, num_pix_y) = self._image_data.shape
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x

        # self._noise_map = data_class.noise_map
        self._sigma_bkg = data_class.background_rms

        if likelihood_mask is None:
            likelihood_mask = np.ones_like(self._image_data)
        self._mask = likelihood_mask
        self._mask_1d = util.image2array(likelihood_mask)

        if psf_class is not None:
            self._psf_kernel = psf_class.kernel_point_source
        else:
            self._psf_kernel = None

        if self._psf_kernel is not None:
            self._conv   = PixelKernelConvolution(self._psf_kernel, convolution_type=convolution_type)
            self._conv_T = PixelKernelConvolution(self._psf_kernel.T, convolution_type=convolution_type)
        else:
            self._conv, self._conv_T = None, None

        self._source_light = source_profile_class
        self._lens_light   = lens_light_profile_class
        if self._lens_light is not None:
            self._solve_for_lens_light = True
        else:
            self._solve_for_lens_light = False

        self._k_max = k_max
        self._n_iter = n_iter
        self._n_weights = n_weights
        # self._scheme = scheme
        # self._tau = tau

        if sparsity_prior_norm not in [0, 1]:
            raise ValueError("Sparsity prior norm can only be 0 or 1 (l0-norm or l1-norm)")
        self._sparsity_prior_norm = sparsity_prior_norm
        self._force_positivity = force_positivity

        self._verbose = verbose


    def solve_sparse(self, lensing_operator_class, kwargs_source, kwargs_lens_light=None):
        if self._solve_for_lens_light:
            # return self._solve_sparse_all(lensing_operator_class, kwargs_source, kwargs_lens_light)
            raise NotImplementedError("SLIT_MCA algorithm not yet implemented")
        else:
            return self._solve_sparse_source(lensing_operator_class, kwargs_source)


    def _solve_sparse_source(self, lensing_operator_class, kwargs_source):
        """SLIT algorithm"""
        self._set_cache(lensing_operator_class, kwargs_source)

        # compute spectral norm of operator "HF." for gradient step
        norm_sp = self.spectral_norm
        mu = 1. / norm_sp  # gradient step

        # get the gradient of the cost function, which is f = || Y - HFS ||^2_2  
        grad_f = lambda x : self.gradient_loss_func(x)

        # get the proximal operator
        prox_g = lambda x, y: self.proximal_sparsity_func(x, y)

        # initial guess as background random noise
        num_pix_source = lensing_operator_class.source_plane_num_pix
        S = self.generate_init_guess(num_pix_source, guess_type='bkg_noise')
        self.quick_imshow(S, title="init", show_now=True)
        # W = 1.

        loss_list = []
        chi2_list = []
        step_diff_list = []
        for j in range(self._n_weights):

            xi, t = 0., 1.
            for i in range(self._n_iter):

                # S_next, xi_next, t_next = algorithms.FISTA_step(S, xi, t, grad_f, prox_g, mu)

                S_next = algorithms.FB_step(S, grad_f, prox_g, mu)

                loss = self.loss_func(S_next)
                chi2 = self.reduced_chi2(S_next)
                step_diff = self.norm_diff(S, S_next)


                if i % 10 == 0:
                    print("iteration {} : loss = {:.4f}, chi2 = {:.4f}, step_diff = {:.4f}"
                          .format(i, loss, chi2, step_diff))

                if i % 30 == 0:
                    self.quick_imshow(S_next, title="iteration {}".format(i), show_now=True)

                loss_list.append(loss)
                chi2_list.append(chi2)
                step_diff_list.append(step_diff)

                # update current estimate of source light
                S = S_next

            # if j == 0:
            #     alpha_0 = alpha.copy()
            # else:
            #     pass

            # W = 2. / ( 1. + np.exp(-10. * (lambda_ - alpha_0)) )


        # save results
        self._source_estimate = S
        self._solve_track = {
            'loss': np.asarray(loss_list),
            'residuals': np.asarray(chi2_list),
            'step_diff': np.asarray(step_diff_list),
        }
        
        # for potential memory issues delete 
        self._unset_cache()
        return self._source_estimate


    def _solve_sparse_all(self, F, kwargs_source, kwargs_lens_light):
        """SLIT_MCA algorithm"""
        pass


    def _set_cache(self, lensing_operator_class, kwargs_source):
        self._lensing_op = lensing_operator_class
        self._kwargs_source = kwargs_source

    def _unset_cache(self):
        delattr(self, '_lensing_op')
        delattr(self, '_kwargs_source')


    @property
    def source_estimate(self):
        if not hasattr(self, '_solve_track'):
            raise ValueError("You must run the optimization before accessing the source estimate")
        return self._source_estimate


    @property
    def solve_track(self):
        if not hasattr(self, '_solve_track'):
            raise ValueError("You must run the optimization before accessing the track")
        return self._solve_track


    def plot_results(self, image_residuals=False):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        ax = axes[0, 0]
        if image_residuals:
            im = ax.imshow(self.normlized_residuals(self.source_estimate), origin='lower')
        else:
            im = ax.imshow(self.source_estimate, origin='lower')
        plot_util.nice_colorbar(im)
        ax = axes[0, 1]
        ax.set_title("loss function")
        ax.plot(self.solve_track['loss'])
        ax.set_xlabel("iterations")
        ax = axes[1, 0]
        ax.set_title("reduced chi2")
        ax.plot(self.solve_track['residuals'])
        ax.set_xlabel("iterations")
        ax = axes[1, 1]
        ax.set_title("step-to-step difference")
        ax.plot(self.solve_track['step_diff'])
        ax.set_xlabel("iterations")
        plt.show()



    @property
    def Y(self):
        """replace masked pixels with random gaussian noise"""
        if not hasattr(self, '_Y'):
            image_data = self._image_data.copy()
            noise = self._sigma_bkg * np.random.randn(*image_data.shape)
            image_data[self._mask] = noise[self._mask]
            self._Y = image_data
        return self._Y


    def generate_init_guess(self, num_pix, guess_type='bkg_noise'):
        if guess_type == 'null':
            return np.zeros((num_pix, num_pix))
        elif guess_type == 'bkg_noise':
            return self._sigma_bkg * np.random.randn(num_pix, num_pix)
        else:
            raise ValueError("Initial guess type '{}' not supported".format(guess_type))



    def apply_mask(self, image_2d):
        return image_2d[self._mask]


    def H(self, array_2d):
        if self._conv is None:
            return array_2d
        return self._conv.convolution2d(array_2d)


    def H_T(self, array_2d):
        if self._conv_T is None:
            return array_2d
        return self._conv_T.convolution2d(array_2d)


    def F(self, source_2d):
        return self._lensing_op.source2image_2d(source_2d)


    def F_T(self, image_2d):
        return self._lensing_op.image2source_2d(image_2d)


    def Phi(self, array_2d):
        return self._source_light.function(array_2d, **self._kwargs_source)


    def Phi_T(self, array_2d):
        return self._source_light.decomposition(array_2d, **self._kwargs_source)


    def loss_func(self, S):
        """ returns f = || Y - HFS ||^2_2 """
        model = self.H(self.F(S))
        error = self.Y - model
        return np.linalg.norm(error, ord=2)**2


    def reduced_chi2(self, S):
        return self.loss_func(S) / self._sigma_bkg**2


    def image_plane_normlized_residuals(self, model):
        """ returns || Y - HFS ||^2_2 / sigma^2 """
        error = self.Y - model
        return (model - error)**2 / self._sigma_bkg**2


    def source_plane_normlized_residuals(self, source_model, true_source):
        """ returns || Y - HFS ||^2_2 / sigma^2 """
        error = true_source - source_model
        return (model - error)**2 / self.noise_levels_source_plane**2


    def norm_diff(self, S1, S2):
        """ returns || S1 - S2 ||^2_2 """
        diff = S1 - S2
        return np.linalg.norm(diff, ord=2)**2


    def gradient_loss_func(self, S):
        """ returns the gradient of f = || Y - HFS ||^2_2 """
        model = self.H(self.F(S))
        error = self.Y - model
        grad  = - self.F_T(self.H_T(error))
        return grad


    def proximal_sparsity_func(self, S, step):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |Phi^T S|_0
        or
            g = lambda * |Phi^T S|_1
        """
        if self._sparsity_prior_norm == 0:
            threshold_func = util.hard_threshold
        else:
            threshold_func = util.soft_threshold

        coeffs_S = self.Phi_T(S)
        n_scales = coeffs_S.shape[0]
        threshold_levels = self.noise_levels_source_plane

        # apply threshold operation to all starlet scales except the coarsest
        for l in range(n_scales-1):
            coeffs_scale_l = coeffs_S[l, :, :]
            levels_scale_l = threshold_levels[l, :, :]
            coeffs_scale_l = threshold_func(coeffs_scale_l, step * levels_scale_l)
            coeffs_S[l, :, :] = coeffs_scale_l

        prox_S = self.Phi(coeffs_S)
        if self._force_positivity:
            prox_S[prox_S < 0] = 0.
        return prox_S


    @property
    def spectral_norm(self):
        if not hasattr(self, '_spectral_norm'):
            def _operator(x):
                x = self.H_T(x)
                x = self.F_T(x)
                x = self.Phi_T(x)
                return x

            def _inverse_operator(x):
                x = self.Phi(x)
                x = self.F(x)
                x = self.H(x)
                return x

            self._spectral_norm = util.spectral_norm(self._num_pix, _operator, _inverse_operator,
                                                     num_iter=20, tol=1e-10)
        return self._spectral_norm


    # def power_method_op(A, A_T, N):
    #     x = np.random.randn(N, 1);
    #     for _ in range(25):
    #        x = x / np.linalg.norm(x, 2)
    #        x = A_T(A(x))
    #     return np.linalg.norm(x, 2)


    # @property
    # def sigma_background_mad(self):
    #     noise_map = self.error_response
    #     mad = np.median(np.abs(noise_map - np.median(noise_map)))
    #     return 1.48 * mad


    @property
    def noise_levels_source_plane(self):
        if not hasattr(self, '_noise_levels_src'):
            self._noise_levels_src = self._compute_noise_levels_src()
        return self._noise_levels_src


    def _compute_noise_levels_src(self):
        n_img = self._lensing_op.image_plane_num_pix

        # estimate noise level
        # sigma_bkg = self.sigma_background_mad(self._noise_map)

        # PSF noise map
        HT = self._psf_kernel.T
        HT_noise = np.ones((n_img, n_img)) * self._sigma_bkg * np.sqrt(np.sum(HT**2))
        FT_HT_noise = self.F_T(HT_noise)
        FT_HT_noise[FT_HT_noise == 0] = 10 * np.mean(FT_HT_noise)

        # computes noise levels in in source plane in starlet space
        dirac = self.dirac_impulse(n_img)
        dirac_mapped = self.F_T(dirac)

        # model transform of the impulse
        dirac_coeffs = self.Phi_T(dirac_mapped)

        noise_levels = np.zeros(dirac_coeffs.shape)
        for scale_idx in range(noise_levels.shape[0]):
            dirac_scale = dirac_coeffs[scale_idx, :, :]
            levels = scp.fftconvolve(FT_HT_noise**2, dirac_scale**2, mode='same')
            levels[levels == 0] = 0
            noise_levels[scale_idx, :, :] = np.sqrt(np.abs(levels))
        return noise_levels


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



    @staticmethod
    def quick_imshow(image, title=None, show_now=False, **kwargs):
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        ax = axes
        if title is not None:
            ax.set_title(title)
        im = ax.imshow(image, origin='lower', **kwargs)
        plot_util.nice_colorbar(im)
        if show_now:
            plt.show()
