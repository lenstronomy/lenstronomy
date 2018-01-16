__author__ = 'sibirrer'

import scipy.optimize
import scipy.interpolate as interpolate
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class SolverUtil(object):
    """
    util functions
    """
    def __init__(self, z_d, z_s):
        self.z_d = z_d
        self.z_s = z_s

    def cosmo2Dd_Ds_Dds(self, H_0, omega_m):
        """

        :param H_0: Hubble constant
        :param omega_m: matter density
        :return: angular diameter distances Dd and Ds/Dds
        """
        cosmo = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=0.)
        lensCosmo = LensCosmo(z_lens=self.z_d, z_source=self.z_s, cosmo=cosmo)
        Dd = lensCosmo.D_d
        Ds = lensCosmo.D_s
        Dds = lensCosmo.D_ds
        return Dd, Ds/Dds


class SolverFlatCosmo(SolverUtil):
    """
    class to solve multidimensional non-linear equations to determine the cosmological parameters H0 and omega_m given
    the angular diameter distance relations
    """
    def F(self, x, Dd, Ds_Dds):
        """

        :param x: array of parameters (H_0, omega_m)
        :return:
        """
        [H_0, omega_m] = x
        omega_m = abs(omega_m)%1
        Dd_new, Ds_Dds_new = self.cosmo2Dd_Ds_Dds(H_0, omega_m)
        y = np.zeros(2)
        y[0] = Dd - Dd_new
        y[1] = Ds_Dds - Ds_Dds_new
        return y

    def solve(self, init, Dd, Ds_Dds):
        x = scipy.optimize.fsolve(self.F, init, args=(Dd, Ds_Dds), xtol=1.49012e-08, factor=0.1)
        x[1] = abs(x[1])%1
        y = self.F(x, Dd, Ds_Dds)
        if abs(y[0]) >= 0.1 or abs(y[1]) > 0.1:
            x = np.array([-1, -1])
        return x


class InvertCosmo(SolverUtil):
    """
    class to do an interpolation and call the inverse of this interpolation to get H_0 and omega_m
    """
    def _make_interpolation(self):
        """
        creates an interpolation grid in H_0, omega_m and computes quantities in Dd and Ds_Dds
        :return:
        """
        H0_range = np.linspace(10, 100, 90)
        omega_m_range = np.linspace(0.05, 1, 95)
        grid2d = np.dstack(np.meshgrid(H0_range, omega_m_range)).reshape(-1, 2)
        H0_grid = grid2d[:, 0]
        omega_m_grid = grid2d[:, 1]
        Dd_grid = np.zeros_like(H0_grid)
        Ds_Dds_grid = np.zeros_like(H0_grid)
        for i in range(len(H0_grid)):
            Dd, Ds_Dds = self.cosmo2Dd_Ds_Dds(H0_grid[i], omega_m_grid[i])
            Dd_grid[i] = Dd
            Ds_Dds_grid[i] = Ds_Dds
        self._f_H0 = interpolate.interp2d(Dd_grid, Ds_Dds_grid, H0_grid, kind='linear', copy=False, bounds_error=False, fill_value=-1)
        print("H0 interpolation done")
        self._f_omega_m = interpolate.interp2d(Dd_grid, Ds_Dds_grid, omega_m_grid, kind='linear', copy=False, bounds_error=False, fill_value=-1)
        print("omega_m interpolation done")

    def get_cosmo(self, Dd, Ds_Dds):
        """
        return the values of H0 and omega_m computed with an interpolation
        :param Dd: flat
        :param Ds_Dds: float
        :return:
        """
        if not hasattr(self, '_f_H0') or not hasattr(self, '_f_omega_m'):
            self._make_interpolation()
        H0 = self._f_H0(Dd, Ds_Dds)
        print(H0, 'H0')
        omega_m = self._f_omega_m(Dd, Ds_Dds)
        Dd_new, Ds_Dds_new = self.cosmo2Dd_Ds_Dds(H0[0], omega_m[0])
        if abs(Dd - Dd_new)/Dd > 0.01 or abs(Ds_Dds - Ds_Dds_new)/Ds_Dds > 0.01:
            return [-1], [-1]
        else:
            return H0[0], omega_m[0]