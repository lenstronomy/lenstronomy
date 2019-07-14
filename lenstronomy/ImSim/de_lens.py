__author__ = 'sibirrer'

import numpy as np
import sys


def get_param_WLS(A, C_D_inv, d, inv_bool=True):
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
        if np.linalg.cond(M) < 5/sys.float_info.epsilon:
            try:
                M_inv = np.linalg.inv(M)
            except:
                M_inv = np.zeros_like(M)
        else:
            M_inv = np.zeros_like(M)
        R = A.T.dot(np.multiply(C_D_inv, d))
        B = M_inv.dot(R)
    else:
        if np.linalg.cond(M) < 5/sys.float_info.epsilon:
            R = A.T.dot(np.multiply(C_D_inv, d))
            try:
                B = np.linalg.solve(M, R).T
            except:
                B = np.zeros(len(A.T))
        else:
            B = np.zeros(len(A.T))
        M_inv = None
    image = A.dot(B)
    return B, M_inv, image


def marginalisation_const(M_inv):
    """
    get marginalisation constant 1/2 log(M_beta) for flat priors
    :param M_inv: 2D covariance matrix
    :return: float
    """

    sign, log_det = np.linalg.slogdet(M_inv)
    if sign == 0:
        return -10**15
    return sign * log_det/2


def marginalization_new(M_inv, d_prior=None):
    """

    :param M_inv: 2D covariance matrix
    :param d_prior: maximum prior length of linear parameters
    :return: log determinant with eigenvalues to be smaller or equal d_prior
    """
    if d_prior is None:
        return marginalisation_const(M_inv)
    v, w = np.linalg.eig(M_inv)
    sign_v = np.sign(v)
    v_abs = np.abs(v)
    #print(np.diag(M_inv), M_inv.max())
    #print(v, v.max())
    v_abs[v_abs > d_prior**2] = d_prior**2
    log_det = np.sum(np.log(v_abs)) * np.prod(sign_v)
    if np.isnan(log_det):
        return -10**15
    m = len(v)
    return log_det / 2 + m/2. * np.log(np.pi/2.) - m * np.log(d_prior)
