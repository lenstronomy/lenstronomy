__author__ = 'aymgal'

import numpy as np

from lenstronomy.Util import util
from lenstronomy.LightModel.Profiles import starlets_slit


class Starlets(object):
    """

    """
    param_names = ['n_scales']
    lower_limit_default = {'n_scales': 2}
    upper_limit_default = {'n_scales': 20}

    def __init__(self, thread_count=1, fast_inverse=True, second_gen=False, show_pysap_plots=False):
        self._use_pysap = self._load_pysap()
        if self._use_pysap:
            self._transf_class = self._pysap.load_transform('BsplineWaveletTransformATrousAlgorithm')
        self._thread_count = thread_count
        self._fast_inverse = fast_inverse
        self._second_gen = second_gen
        self._show_pysap_plots = show_pysap_plots

    def function(self, coeffs, n_scales):
        """return inverse starlet transform from starlet coefficients stored in amp"""
        if self._use_pysap:
            return self._inverse_transform(coeffs, n_scales)
        else:
            return starlets_slit._inverse_transform(coeffs, fast=self._fast_inverse, 
                                                    second_gen=self._second_gen)

    def decomposition(self, image, n_scales):
        """
        decomposes an image into starlet coefficients
        :return:
        """
        if self._use_pysap:
            return self._transform(image, n_scales)
        else:
            return starlets_slit._transform(image, n_scales, second_gen=self._second_gen)


    def spectral_norm(self, num_pix, n_scales):
        if not hasattr(self, '_spectral_norm') or n_scales != self._n_scales_cache:
            self._spectral_norm = self._compute_spectral_norm(num_pix, n_scales, num_iter=20, tol=1e-10)
            self._n_scales_cache = n_scales
        return self._spectral_norm


    def _inverse_transform(self, coeffs, n_scales):
        """performs inverse starlet transform"""
        self._check_transform_pysap(n_scales)
        if self._fast_inverse and not self._second_gen:
            # for 1st gen starlet the reconstruction can be performed by summing all scales 
            image = np.sum(coeffs, axis=0)
        else:
            coeffs = self._coeffs2pysap(coeffs)
            self._transf.analysis_data = coeffs
            result = self._transf.synthesis()
            if self._show_pysap_plots:
                result.show()
            image = result.data
        return image

    def _transform(self, image, n_scales):
        """
        decomposes an image into starlets coefficients
        """
        self._check_transform_pysap(n_scales)
        self._transf.data = image
        self._transf.analysis()
        if self._show_pysap_plots:
            self._transf.show()
        coeffs = self._transf.analysis_data
        coeffs = self._pysap2coeffs(coeffs)
        return coeffs

    def _check_transform_pysap(self, n_scales):
        if not hasattr(self, '_transf') or n_scales != self._n_scales:
            self._transf = self._transf_class(nb_scale=n_scales, verbose=False, 
                                              nb_procs=self._thread_count)
            self._n_scales = n_scales

    def _pysap2coeffs(self, coeffs):
        return np.asarray(coeffs)

    def _coeffs2pysap(self, coeffs):
        coeffs_list = []
        for i in range(coeffs.shape[0]):
            coeffs_list.append(coeffs[i, :, :])
        return coeffs_list

    def _compute_spectral_norm(self, num_pix, n_scales, num_iter=20, tol=1e-10):
        """compute spectral norm of the starlet operator"""
        operator = lambda x: self.decomposition(x, n_scales)
        inverse_operator = lambda c: self.function(c, n_scales)
        return util.spectral_norm(num_pix, operator, inverse_operator, num_iter=num_iter, tol=tol)


    def _load_pysap(self):
        try:
            import pysap
        except ImportError:
            self._pysap = None
            return False
        else:
            self._pysap = pysap
            return True