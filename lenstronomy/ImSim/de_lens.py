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
        if np.linalg.cond(M) < 10/sys.float_info.epsilon:
            try:
                M_inv = np.linalg.inv(M)
            except:
                M_inv = np.zeros_like(M)
        else:
            M_inv = np.zeros_like(M)
        R = A.T.dot(np.multiply(C_D_inv, d))
        B = M_inv.dot(R)
    else:
        if np.linalg.cond(M) < 10/sys.float_info.epsilon:
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
    return log_det/2


