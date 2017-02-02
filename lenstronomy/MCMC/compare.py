__author__ = 'sibirrer'

import numpy as np

class Compare(object):
    """
    this class is aimed to compare two images
    """
    def __init__(self, kwargs_options):
        """

        :param kwargs_options: options for the comparison tool
        :param mask_array: if not == None, this is a mask which hides part of the real image and only compares a fraction of its pixels
        :return:
        """
        self.kwargs_options = kwargs_options

    def residual_map(self, model, data, sigma, reduce_frac=1, mask=1, model_error=0):
        """
        returns reduced residual map
        :param model:
        :param data:
        :param sigma:
        :param reduce_frac:
        :param mask:
        :param model_error:
        :return:
        """
        self.check_comparability(model, data)
        deltaIm = (data-model)**2*mask
        model_pos = np.empty_like(model)
        model[np.isnan(model)] = 0
        model_pos[model >= 0] = model[model >= 0]
        model_pos[model < 0] = 0

        relDeltaIm = deltaIm/(sigma**2 + model_pos/reduce_frac + model_error)
        return relDeltaIm, deltaIm

    def compare2D(self, model, data, sigma, reduce_frac=1, mask=1, model_error=0):
        """

        :param model: model 2d image
        :param data: data 2d image
        :param sigma: minimal noise level of background (float>0 or as image)
        :return: X^2 value if images have same size
        """
        relDeltaIm, deltaIm = self.residual_map(model, data, sigma, reduce_frac, mask, model_error)
        if self.kwargs_options['X2_compare'] == 'simple':
            X2_simple = np.sum(deltaIm)
            return X2_simple
        elif self.kwargs_options['X2_compare'] == 'standard':
            X2_estimate = np.sum(relDeltaIm)
            return X2_estimate
        else:
            raise ValueError('kwargs_options has invalid keyword %s' %(self.kwargs_options['X2_compare']))

    def compare_distance(self, x_mapped, y_mapped):
        """

        :param x_mapped: array of x-positions of remapped catalogue image
        :param y_mapped: array of y-positions of remapped catalogue image
        :return: sum of distance square of positions
        """
        X2 = 0
        for i in range(0,len(x_mapped)-1):
            for j in range(i+1,len(x_mapped)):
                dx = x_mapped[i]-x_mapped[j]
                dy = y_mapped[i]-y_mapped[j]
                X2 += dx**2+dy**2
        return X2

    def delays(self, delays_model, delays_measured, delays_errors):
        """

        :param delays_model: 4 delays of the model (not relative delays)
        :param delays_measured: relative delays (1-2,1-3,1-4)
        :param delays_errors: gaussian errors on the measured delays
        :return: log likelihood of data given model
        """
        if len(delays_model) != 4:
            raise ValueError('time delay information not compatible!')
        dt1 = delays_model[1] - delays_model[0]
        dt2 = delays_model[2] - delays_model[0]
        dt3 = delays_model[3] - delays_model[0]
        dt = np.array([dt1, dt2, dt3])
        logL = np.sum(-(dt - delays_measured)**2/(2*delays_errors**2))
        return logL

    def check_comparability(self, image1, image2):
        """

        :param image1: 2d image
        :param image2: 2d image
        :return: True if same size, raises error if not
        """
        if (not len(image1) == len(image2)) or (not np.size(image1) == np.size(image2)):
            raise ValueError('image 1 has not the same pixels as image 2. Refused to compare!')
        return True

    def get_marg_const(self, M_inv):
        """
        get marginalisation constant 1/2 log(M_beta)
        :param M_inv: 2D covariance matrix
        :return: float
        """
        sign, log_det = log_det_N_inv = np.linalg.slogdet(M_inv)
        return log_det/2

    def get_log_likelihood(self, model, data, sigma, reduce_frac=1, mask=1, model_error=0, cov_matrix=None):
        X2 = self.compare2D(model, data, sigma, reduce_frac, mask, model_error)
        X2 /= 2 # from chi^2 to log likelihood
        if cov_matrix is not None and self.kwargs_options.get('source_marg', False):
            print('this should not happen!!!!')
            marg_const = self.get_marg_const(cov_matrix)
            if marg_const + X2 > 0:
                X2 += marg_const
        return -X2
