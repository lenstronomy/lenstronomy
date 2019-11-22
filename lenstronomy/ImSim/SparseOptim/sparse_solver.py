__author__ = 'aymgal'

# class that implements SLIT algorithm

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from lenstronomy.Util import util
from lenstronomy.Plots import plot_util
from lenstronomy.ImSim.SparseOptim import algorithms
from lenstronomy.ImSim.SparseOptim import proximals



class SparseSolver(object):


    def __init__(self, data_class, source_profile_class, psf_class=None, lens_light_profile_class=None, likelihood_mask=None, 
                 k_max=5, n_iter=50, n_weights=1, sparsity_prior_norm=1, force_positivity=True, 
                 formulation='analysis', convolution_type='fft_static', verbose=False, show_steps=False):

        self._image_data = data_class.data

        (num_pix_x, num_pix_y) = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._delta_pix = data_class.pixel_width

        # self._noise_map = data_class.noise_map
        self._sigma_bkg = data_class.background_rms

        if likelihood_mask is None:
            likelihood_mask = np.ones_like(self._image_data)
        self._mask = likelihood_mask
        self._mask_1d = util.image2array(likelihood_mask)

        if psf_class is not None:
            self._psf_kernel = psf_class.kernel_point_source
            self._conv   = PixelKernelConvolution(self._psf_kernel, convolution_type=convolution_type)
            self._conv_T = PixelKernelConvolution(self._psf_kernel.T, convolution_type=convolution_type)
        else:
            self._psf_kernel = None
            self._conv, self._conv_T = None, None

        self._source_light = source_profile_class
        self._lens_light   = lens_light_profile_class
        if self._lens_light is None:
            self._solve_for_lens_light = False
        else:
            self._solve_for_lens_light = True

        self._formulation = formulation
        self._k_max = k_max
        self._n_iter = n_iter
        self._n_weights = n_weights
        # self._tau = tau

        if sparsity_prior_norm not in [0, 1]:
            raise ValueError("Sparsity prior norm can only be 0 or 1 (l0-norm or l1-norm)")
        self._sparsity_prior_norm = sparsity_prior_norm
        self._force_positivity = force_positivity

        self._verbose = verbose
        self._show_steps = show_steps


    def solve(self, lensing_operator_class, kwargs_source, kwargs_lens_light=None):
        if self._solve_for_lens_light:
            # return self._solve_all(lensing_operator_class, kwargs_source, kwargs_lens_light)
            raise NotImplementedError("Sparse solver for source and lens light not implemented")
        else:
            return self._solve_source(lensing_operator_class, kwargs_source)


    def _solve_source(self, lensing_operator_class, kwargs_source):
        """SLIT algorithm"""
        self._set_cache(lensing_operator_class, kwargs_source)

        # compute the gradient step
        mu = 1. / self.spectral_norm

        # get the gradient of the cost function, which is f = || Y - HFS ||^2_2  
        grad_f = lambda x : self.gradient_loss(x)

        # initial guess as background random noise
        num_pix_source = lensing_operator_class.source_plane_num_pix
        S, alpha_S = self.generate_initial_guess(num_pix_source, kwargs_source['n_scales'],
                                                 guess_type='bkg_noise')
        if self._show_steps:
            self.quick_imshow(S, title="initial guess", show_now=True)

        # initialise weights
        weights = 1.

        loss_list = []
        red_chi2_list = []
        step_diff_list = []
        for j in range(self._n_weights):

            if j == 0 and self.algorithm == 'FISTA':
                fista_xi = np.copy(alpha_S)
                fista_t  = 1.

            for i in range(self._n_iter):

                # get the proximal operator at current step
                prox_g = lambda x, y: self.proximal_sparsity(x, y, weights)

                if self.algorithm == 'FISTA':
                    alpha_S_next, fista_xi_next, fista_t_next \
                        = algorithms.step_FISTA(alpha_S, fista_xi, fista_t, grad_f, prox_g, mu)
                    S_next = self.Phi(alpha_S_next)

                elif self.algorithm == 'FB':
                    S_next = algorithms.step_FB(S, grad_f, prox_g, mu)

                loss = self.loss(S_next)
                red_chi2 = self.reduced_chi2(S_next)
                step_diff = self.norm_diff(S, S_next)

                if i % 10 == 0 and self._verbose:
                    print("iteration {}-{} : loss = {:.4f}, red-chi2 = {:.4f}, step_diff = {:.4f}"
                          .format(j, i, loss, red_chi2, step_diff))

                if i % int(self._n_iter/2) == 0 and self._show_steps:
                    self.quick_imshow(S_next, title="iteration {}".format(i), show_now=True, cmap='gist_stern')

                loss_list.append(loss)
                red_chi2_list.append(red_chi2)
                step_diff_list.append(step_diff)

                # update current estimate of source light
                S = S_next
                if self.algorithm == 'FISTA':
                    alpha_S = alpha_S_next
                    fista_xi, fista_t = fista_xi_next, fista_t_next

            if j == 0:
                # save coefficients from first inner loop estimate
                alpha_0 = np.copy(alpha_S)

            # update weights
            lambda_ = self._k_max * self.noise_levels_source_plane
            weights = 2. / ( 1. + np.exp(-10. * (lambda_ - alpha_0)) )

        # save results
        self._source_model = S
        self._solve_track = {
            'loss': np.asarray(loss_list),
            'red_chi2': np.asarray(red_chi2_list),
            'step_diff': np.asarray(step_diff_list),
        }

        if self._show_steps:
            self.quick_imshow(S, title="final estimate", show_now=True, cmap='gist_stern')
        
        # for potential memory issues delete 
        # self._unset_cache()
        return self._source_model


    def _solve_all(self, F, kwargs_source, kwargs_lens_light):
        """SLIT_MCA algorithm"""
        pass


    def _set_cache(self, lensing_operator_class, kwargs_source):
        self._lensing_op = lensing_operator_class
        self._kwargs_source = kwargs_source

    def _unset_cache(self):
        delattr(self, '_lensing_op')
        delattr(self, '_kwargs_source')


    @property
    def source_model(self):
        if not hasattr(self, '_source_model'):
            raise ValueError("You must run the optimization before accessing the source estimate")
        return self._source_model


    def image_model(self, unconvolved=False):
        if not hasattr(self, '_source_model'):
            raise ValueError("You must run the optimization before accessing the source estimate")
        image_model = self.F(self._source_model)
        if unconvolved:
            return image_model
        return self.H(image_model)


    @property
    def solve_track(self):
        if not hasattr(self, '_solve_track'):
            raise ValueError("You must run the optimization before accessing the track")
        return self._solve_track


    @property
    def best_fit_reduced_chi2(self):
        if not hasattr(self, '_solve_track'):
            raise ValueError("You must run the optimization before accessing the track")
        return self._solve_track['red_chi2'][-1]


    def plot_results(self, image_residuals=False, vmin=None, vmax=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        if image_residuals:
            ax.set_title(r"$(data - model)/\sigma^2$")
            im = ax.imshow(self.reduced_residuals(self.source_model), origin='lower',
                           cmap='bwr', vmin=vmin, vmax=vmax)
            frame_size = self._num_pix * self._delta_pix
            text = r"$\chi^2={:.2f}$".format(self.best_fit_reduced_chi2)
            plot_util.text_description(ax, frame_size, text, color='black', backgroundcolor='white',
                                       flipped=False, font_size=15)
        else:
            im = ax.imshow(self.image_model(unconvolved=False), origin='lower', cmap='cubehelix')
        plot_util.nice_colorbar(im)
        ax = axes[0, 1]
        ax.set_title("loss function")
        ax.plot(self.solve_track['loss'])
        ax.set_xlabel("iterations")
        ax = axes[1, 0]
        ax.set_title("reduced chi2")
        ax.plot(self.solve_track['red_chi2'])
        ax.set_xlabel("iterations")
        ax = axes[1, 1]
        ax.set_title("step-to-step difference")
        ax.semilogy(self.solve_track['step_diff'])
        ax.set_xlabel("iterations")
        plt.show()


    def generate_initial_guess(self, num_pix, n_scales, guess_type='bkg_noise'):
        if guess_type == 'null':
            X = np.zeros((num_pix, num_pix))
            alpha_X = np.zeros((n_scales, num_pix, num_pix))
        elif guess_type == 'bkg_noise':
            if self._formulation == 'analysis':
                X = self._sigma_bkg * np.random.randn(num_pix, num_pix)
                alpha_X = self.Phi_T(X)
            elif self._formulation == 'synthesis':
                sigma_bkg = self.noise_levels_source_plane
                alpha_X = sigma_bkg * np.random.randn(num_pix, num_pix)
                X = self.Phi(alpha_X)
        else:
            raise ValueError("Initial guess type '{}' not supported".format(guess_type))
        return X, alpha_X


    def apply_mask(self, image_2d):
        # image_2d_m = image_2d.copy()
        # image_2d_m[self._mask] = 0.
        # return image_2d_m
        return image_2d * self._mask


    def apply_source_plane_mask(self, source_2d):
        return source_2d * self.mask_source_plane


    @property
    def image_data(self):
        return self._image_data


    @property
    def Y(self):
        """replace masked pixels with random gaussian noise"""
        if not hasattr(self, '_Y'):
            image_data = np.copy(self._image_data)
            noise = self._sigma_bkg * np.random.randn(*image_data.shape)
            image_data[~self._mask] = noise[~self._mask]
            self._Y = image_data
        return self._Y


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


    @property
    def num_data_evaluate(self):
        """
        number of data points to be used in the linear solver
        :return:
        """
        return int(np.sum(self._mask))


    def loss(self, S):
        """ returns f = || Y - HFS ||^2_2 """
        model = self.H(self.F(S))
        error = self.Y - model
        norm_error = np.linalg.norm(error, ord=2)
        return norm_error**2


    def reduced_chi2(self, S):
        chi2 = self.reduced_residuals(S)**2
        return np.sum(chi2) / self.num_data_evaluate


    def reduced_residuals(self, S):
        """ returns || Y - HFS ||^2_2 / sigma^2 """
        model = self.H(self.F(S))
        error = self.Y - model
        return error / self._sigma_bkg


    def norm_diff(self, S1, S2):
        """ returns || S1 - S2 ||^2_2 """
        diff = S1 - S2
        return np.linalg.norm(diff, ord=2)**2


    def gradient_loss(self, array):
        if self._formulation == 'analysis':
            return self._gradient_loss_analysis(array)
        elif self._formulation == 'synthesis':
            return self._gradient_loss_synthesis(array)


    def _gradient_loss_analysis(self, S):
        """ returns the gradient of f = || Y - HFS ||^2_2 """
        model = self.H(self.F(S))
        error = self.Y - model
        grad  = - self.F_T(self.H_T(error))
        return grad


    def _gradient_loss_synthesis(self, alpha_S):
        """ returns the gradient of f = || Y - H F Phi alphaS ||^2_2 """
        model = self.H(self.F(self.Phi(alpha_S)))
        error = self.Y - model
        grad  = - self.Phi_T(self.F_T(self.H_T(error)))
        return grad


    def proximal_sparsity(self, array, step, weights):
        if self._formulation == 'analysis':
            return self._proximal_sparsity_analysis(array, step, weights)
        elif self._formulation == 'synthesis':
            return self._proximal_sparsity_synthesis(array, step, weights)


    def _proximal_sparsity_analysis(self, S, step, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |Phi^T S|_0
        or
            g = lambda * |Phi^T S|_1
        """
        n_scales = self.noise_levels_source_plane.shape[0]
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max + 1  # means a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_source_plane

        alpha_S = self.Phi_T(S)
        alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step, level_const=level_const, level_pixels=level_pixels,
                                                          force_positivity=self._force_positivity, norm=self._sparsity_prior_norm)
        S_proxed = self.Phi(alpha_S_proxed)

        if self._force_positivity:
            S_proxed = proximals.prox_positivity(S_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        S_proxed = self.apply_source_plane_mask(S_proxed)
        return S_proxed


    def _proximal_sparsity_synthesis(self, alpha_S, step, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |alpha_S|_0
        or
            g = lambda * |alpha_S|_1
        """
        n_scales = self.noise_levels_source_plane.shape[0]
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max + 1  # means a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_source_plane

        alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step, level_const=level_const, level_pixels=level_pixels,
                                                          force_positivity=self._force_positivity, norm=self._sparsity_prior_norm)

        if self._force_positivity:
            alpha_S_proxed = proximals.prox_positivity(alpha_S_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        alpha_S_proxed = self.apply_source_plane_mask(alpha_S_proxed)
        return alpha_S_proxed


    @property
    def algorithm(self):
        if self._formulation == 'analysis':
            return 'FB'
        elif self._formulation == 'synthesis':
            return 'FISTA'


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
    def mask_source_plane(self):
        # TODO : include mask in image plane ray-traced to source plane
        return self._lensing_op.source_support


    @property
    def noise_levels_source_plane(self):
        if not hasattr(self, '_noise_levels_src'):
            self._noise_levels_src = self._compute_noise_levels_src()
        return self._noise_levels_src


    def _compute_noise_levels_src(self):
        n_img = self._lensing_op.image_plane_num_pix

        # PSF noise map
        HT = self._psf_kernel.T
        HT_power = np.sqrt(np.sum(HT**2))
        HT_noise = self._sigma_bkg * HT_power * np.ones((n_img, n_img))
        FT_HT_noise = self.F_T(HT_noise)
        FT_HT_noise[FT_HT_noise == 0.] = 10. * np.mean(FT_HT_noise)

        # computes noise levels in in source plane in starlet space
        dirac = self.dirac_impulse(n_img)
        dirac_mapped = self.F_T(dirac)

        # model transform of the impulse
        dirac_coeffs = self.Phi_T(dirac_mapped)

        noise_levels = np.zeros(dirac_coeffs.shape)
        for scale_idx in range(noise_levels.shape[0]):
            dirac_scale = dirac_coeffs[scale_idx, :, :]
            levels = signal.fftconvolve(FT_HT_noise**2, dirac_scale**2, mode='same')
            levels[levels == 0] = 0
            noise_levels[scale_idx, :, :] = np.sqrt(np.abs(levels))
        return noise_levels


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
