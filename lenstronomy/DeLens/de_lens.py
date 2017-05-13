__author__ = 'sibirrer'

import numpy as np
import sys

import astrofunc.util as util
from astrofunc.LensingProfiles.shapelets import Shapelets

class DeLens(object):
    """
    class for the de-lensing algorithm
    """

    def get_param_WLS(self, A, C_D_inv, d, inv_bool=True):
        """
        returns the parameter values given
        :param A: response matrix Nd x Ns (Nd = # data points, Ns = # parameters)
        :param C_D_inv: inverse covariance matrix of the data, Nd x Nd, diagonal form
        :param d: data array, 1-d Nd
        :param inv_bool: boolean, wheter returning also the inverse matrix or just solve the linear system
        :return: 1-d array of parameter values
        """
        M = A.T.dot(np.multiply(C_D_inv, A.T).T)

        if inv_bool:
            if np.linalg.cond(M) < 1/sys.float_info.epsilon:
                M_inv = np.linalg.inv(M)
            else:
                M_inv = np.zeros_like(M)
            R = A.T.dot(np.multiply(C_D_inv, d))
            B = M_inv.dot(R)
        else:
            if np.linalg.cond(M) < 10/sys.float_info.epsilon:
                R = A.T.dot(np.multiply(C_D_inv, d))
                B = np.linalg.solve(M, R).T
            else:
                B = np.zeros(len(A.T))
            M_inv = None
        image = A.dot(B)
        return B, M_inv, image

    def get_covariance_matrix(self, d, sigma_b, f):
        """
        returns a diagonal matrix for the covariance estimation
        :param d: data array
        :param sigma_b: background noise
        :param f: reduced poissonian noise
        :return: len(d) x len(d) matrix
        """
        if isinstance(f, int) or isinstance(f, float):
            if f <= 0:
                f = 1
        else:
            mean_exp_time = np.mean(f)
            f[f < mean_exp_time / 10] = mean_exp_time / 10

        if sigma_b * np.max(f) < 1:
            print("WARNING! sigma_b*f %s >1 may introduce unstable error estimates" % (sigma_b*np.max(f)))
        d_pos = np.zeros_like(d)
        #threshold = 1.5*sigma_b
        d_pos[d >= 0] = d[d >= 0]
        #d_pos[d < threshold] = 0
        sigma = d_pos/f + sigma_b**2
        return sigma

    def get_source(self, param, num_order, beta, center_x, center_y, x_grid, y_grid):
        """

        :param makeImage:
        :param param:
        :param num_order:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        shapelets = Shapelets(interpolation=False, precalc = False)
        source = np.zeros(len(x_grid))

        n1 = 0
        n2 = 0
        for i in range(0,len(param)):
            source += shapelets.function(x_grid, y_grid, param[i], beta, n1, n2, center_x, center_y)
            if n1 + n2 < num_order:
                n1 += 1
            else:
                n1 = 0
                n2 += 1
        return source


    def get_response_matrix_new(self, makeImage, makeImage_lens, num_order, x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res, center_x, center_y, beta, mask):
        """

        :param makeImage: instance of a class makeImage with shapelets
        :param x_grid:
        :param y_grid:
        :param num_param:
        :param kwargs_lens:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = (num_order+2)*(num_order+1)/2 + 2
        A = np.zeros((num_param, numPix**2))
        print num_param, 'num_param'
        print np.shape(A)
        n1 = 0
        n2 = 0
        kwargs_else_point = {'point_amp':1}
        point_source = makeImage_lens.make_image_point_source(kwargs_lens, kwargs_source, kwargs_psf, kwargs_else_point, numPix, deltaPix)
        A[0,:] = util.image2array(point_source)
        kwargs_source_extended = kwargs_source
        kwargs_source_extended['I0_sersic'] = 1.
        extended_source = makeImage_lens.make_image_surface_extended_source(x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res)
        A[1,:] = util.image2array(extended_source)
        for i in range(2,num_param):
            kwargs_source = {'center_x':center_x, 'center_y':center_y, 'n1':n1, 'n2':n2, 'beta':beta, 'amp':1}
            image = makeImage.make_image_surface_extended_source(x_grid, y_grid, kwargs_lens, kwargs_source, kwargs_psf, kwargs_else, numPix, deltaPix, subgrid_res)
            image_masked = image*mask
            response = util.image2array(image_masked)

            A[i,:] = response
            if n1 + n2 < num_order:
                n1 += 1
            else:
                n1 = 0
                n2 += 1
        return A

    def get_source_new(self, param, num_order, beta, center_x, center_y, x_grid, y_grid):
        """

        :param makeImage:
        :param param:
        :param num_order:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        shapelets = Shapelets(interpolation=False, precalc = False)
        source = np.zeros(len(x_grid))

        n1 = 0
        n2 = 0
        for i in range(1,len(param)):
            source += shapelets.function(x_grid, y_grid, param[i], beta, n1, n2, center_x, center_y)
            if n1 + n2 < num_order:
                n1 += 1
            else:
                n1 = 0
                n2 += 1
        return source