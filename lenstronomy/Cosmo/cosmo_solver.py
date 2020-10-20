__author__ = 'sibirrer'

import scipy.optimize
import scipy.interpolate as interpolate
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def cosmo2angular_diameter_distances(H_0, omega_m, z_lens, z_source):
    """

    :param H_0: Hubble constant [km/s/Mpc]
    :param omega_m: dimensionless matter density at z=0
    :param z_lens: deflector redshift
    :param z_source: source redshift
    :return: angular diameter distances Dd and Ds/Dds
    """
    cosmo = FlatLambdaCDM(H0=H_0, Om0=omega_m, Ob0=0.)
    lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    Dd = lensCosmo.dd
    Ds = lensCosmo.ds
    Dds = lensCosmo.dds
    return Dd, Ds/Dds


@export
def ddt2h0(ddt, z_lens, z_source, cosmo):
    """
    converts time-delay distance to H0 for a given expansion history

    :param ddt: time-delay distance in Mpc
    :param z_lens: deflector redshift
    :param z_source: source redshift
    :param cosmo: astropy.cosmology class instance
    :return: h0 value which matches the cosmology class effectively replacing the h0 value used in the creation of this class
    """
    h0_fiducial = cosmo.H0.value
    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    ddt_fiducial = lens_cosmo.ddt
    h0 = h0_fiducial * ddt_fiducial / ddt
    return h0


@export
class SolverFlatLCDM(object):
    """
    class to solve multidimensional non-linear equations to determine the cosmological parameters H0 and omega_m given
    the angular diameter distance relations
    """
    def __init__(self, z_d, z_s):
        self.z_d = z_d
        self.z_s = z_s

    def F(self, x, Dd, Ds_Dds):
        """

        :param x: array of parameters (H_0, omega_m)
        :return:
        """
        [H_0, omega_m] = x
        omega_m = abs(omega_m)%1
        Dd_new, Ds_Dds_new = cosmo2angular_diameter_distances(H_0, omega_m, self.z_d, self.z_s)
        y = np.zeros(2)
        y[0] = Dd - Dd_new
        y[1] = Ds_Dds - Ds_Dds_new
        return y

    def solve(self, init, dd, ds_dds):
        x = scipy.optimize.fsolve(self.F, init, args=(dd, ds_dds), xtol=1.49012e-08, factor=0.1)
        x[1] = abs(x[1]) % 1
        y = self.F(x, dd, ds_dds)
        if abs(y[0]) >= 0.1 or abs(y[1]) > 0.1:
            x = np.array([-1, -1])
        return x


@export
class InvertCosmo(object):
    """
    class to do an interpolation and call the inverse of this interpolation to get H_0 and omega_m
    """
    def __init__(self, z_d, z_s, H0_range=np.linspace(10, 100, 90), omega_m_range=np.linspace(0.05, 1, 95)):
        self.z_d = z_d
        self.z_s = z_s
        self._H0_range = H0_range
        self._omega_m_range = omega_m_range

    def _make_interpolation(self):
        """
        creates an interpolation grid in H_0, omega_m and computes quantities in Dd and Ds_Dds
        :return:
        """
        grid2d = np.dstack(np.meshgrid(self._H0_range, self._omega_m_range)).reshape(-1, 2)
        H0_grid = grid2d[:, 0]
        omega_m_grid = grid2d[:, 1]
        Dd_grid = np.zeros_like(H0_grid)
        Ds_Dds_grid = np.zeros_like(H0_grid)
        for i in range(len(H0_grid)):
            Dd, Ds_Dds = cosmo2angular_diameter_distances(H0_grid[i], omega_m_grid[i], self.z_d, self.z_s)
            Dd_grid[i] = Dd
            Ds_Dds_grid[i] = Ds_Dds
        self._f_H0 = interpolate.interp2d(Dd_grid, Ds_Dds_grid, H0_grid, kind='linear', copy=False, bounds_error=False, fill_value=-1)
        print("H0 interpolation done")
        self._f_omega_m = interpolate.interp2d(Dd_grid, Ds_Dds_grid, omega_m_grid, kind='linear', copy=False, bounds_error=False, fill_value=0)
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
        Dd_new, Ds_Dds_new = cosmo2angular_diameter_distances(H0[0], omega_m[0], self.z_d, self.z_s)
        if abs(Dd - Dd_new)/Dd > 0.01 or abs(Ds_Dds - Ds_Dds_new)/Ds_Dds > 0.01:
            return -1, -1
        else:
            return H0[0], omega_m[0]
