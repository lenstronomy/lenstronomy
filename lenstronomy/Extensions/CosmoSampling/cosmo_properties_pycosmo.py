__author__ = 'sibirrer'

#this files contains a class to manage the basic outputs from a given cosmology

import mpmath as mp

import PyCosmo
import numpy as np
import lenstronomy.Util.constants as constants

class CosmoProp(object):

    def __init__(self, z_lens, z_source, param_file=None):

        #basically we just need the parameter file and z_lens and z_source
        if param_file == None:
            param_file = "Config.pycosmo_config_planck2013"
        self.cosmo = PyCosmo.Cosmo(param_file)
        self.z_L = z_lens
        self.z_S = z_source
        self.a_L = self.a_z(self.z_L)
        self.a_S = self.a_z(self.z_S)

    def a_z(self, z):
        """
        returns scale factor (a_0 = 1) for given redshift
        """
        return 1./(1+z)

    def D_d(self, H0, omega_m):
        self.cosmo.params.h = H0/100.
        self.cosmo.params.H0 = H0
        self.cosmo.params.omega_m = omega_m
        return self.cosmo.background.dist_ang_a(self.a_L)[0]

    def D_s(self, H0, omega_m):
        self.cosmo.params.h = H0/100.
        self.cosmo.params.H0 = H0
        self.cosmo.params.omega_m = omega_m
        return self.cosmo.background.dist_ang_a(self.a_S)[0]

    def D_ds(self, H0, omega_m):
        self.cosmo.params.h = H0/100.
        self.cosmo.params.H0 = H0
        self.cosmo.params.omega_m = omega_m
        return (self.cosmo.background.dist_trans_a(self.a_S)[0] - self.cosmo.background.dist_trans_a(self.a_L)[0])*self.a_S

    def get_angular_distance_H0(self, H0):
        """
        returns arrays of constant H0 with varying omega_m
        :param H0:
        :return:
        """
        n = 101
        omega_m_array = np.linspace(0, 1, n)
        D_d_array = np.zeros_like(omega_m_array)
        D_s_array = np.zeros_like(omega_m_array)
        D_ds_array = np.zeros_like(omega_m_array)

        for i in range(n):
            omega_m = omega_m_array[i]
            D_d_array[i] = self.D_d(H0, omega_m)
            D_s_array[i] = self.D_s(H0, omega_m)
            D_ds_array[i] = self.D_ds(H0, omega_m)
        return D_d_array, D_s_array, D_ds_array

    def get_angular_distance_omega_m(self, omega_m):
        """
        returns arrays of constant H0 with varying omega_m
        :param H0:
        :return:
        """
        n = 201
        H0_array = np.linspace(0, 200, n)
        D_d_array = np.zeros_like(H0_array)
        D_s_array = np.zeros_like(H0_array)
        D_ds_array = np.zeros_like(H0_array)

        for i in range(n):
            H0 = H0_array[i]
            D_d_array[i] = self.D_d(H0, omega_m)
            D_s_array[i] = self.D_s(H0, omega_m)
            D_ds_array[i] = self.D_ds(H0, omega_m)
        return D_d_array, D_s_array, D_ds_array

    @property
    def dist_OL(self):
        """
        returns angular diameter distance in physical Mpc from observer-lens
        """
        if not hasattr(self,'dist_OL_value'):
            self.dist_OL_value = self.cosmo.background.dist_ang_a(self.a_L)[0]
        return self.dist_OL_value

    @property
    def dist_OS(self):
        """
        returns angular diameter distance in physical Mpc from observer-source
        """
        if not hasattr(self,'dist_OS_value'):
            self.dist_OS_value = self.cosmo.background.dist_ang_a(self.a_S)[0]
        return self.dist_OS_value

    @property
    def dist_LS(self):
        """
        returns angular diameter distance in physical Mpc from lens-source
        """
        if not hasattr(self,'dist_LS_value'):
            self.dist_LS_value = (self.cosmo.background.dist_trans_a(self.a_S)[0] - self.cosmo.background.dist_trans_a(self.a_L)[0])*self.a_S
        return self.dist_LS_value

    @property
    def epsilon_crit(self):
        """
        returns the critical projected mass density in units of M_sun/Mpc^2 (physical units)
        """
        if not hasattr(self,'Epsilon_Crit'):
            const_SI = constants.c ** 2 / (4 * np.pi * constants.G)  #c^2/(4*pi*G) in units of [kg/m]
            conversion = constants.Mpc / constants.M_sun  # converts [kg/m] to [M_sun/Mpc]
            const = const_SI*conversion   #c^2/(4*pi*G) in units of [M_sun/Mpc]
            self.Epsilon_Crit = self.dist_OS/(self.dist_OL*self.dist_LS) * const #[M_sun/Mpc^2]
        return self.Epsilon_Crit

    @property
    def D_dt_model(self):
        if not hasattr(self, '_D_dt_model'):
            self._D_dt_model = (1 + self.z_L) * self.dist_OL * self.dist_OS / self.dist_LS
        return self._D_dt_model

    @property
    def hubble_small(self):
        """
        returns little h
        """
        if not hasattr(self, 'h'):
            self.h = self.cosmo.params.h
        return self.h

    def rCom(self, z, Om=1):
        """
        Flat LCDM comoving distance to redshift z, in Mpc/h.
        rCom(z) = int dz / H(z)

        :param z: redshift or array of redshifts
        :param Om (optional): matter density, default: Om in defaultCosmology
        :returns res: comoving distance to z in Mpc/h
        """
        if Om < 1:
            OmbyOlam = Om / (1 - Om)
            out = ((1 + z) * mp.hyp2f1(1/3.0, 0.5, 4/3.0, -(1 + z) * (1 + z) * (1 + z)
                                    * OmbyOlam)
                   - mp.hyp2f1(1/3.0, 0.5, 4/3.0, -OmbyOlam))
            out *= constants.c * 10 / np.sqrt(1 - Om)
        else:
            out = constants.c * 20 * (1 - 1. / mp.sqrt(1 + z))
        return out

    @property
    def rho_crit(self):
        return 3 * self.hubble_small ** 2 / (8 * np.pi * constants.G) * 10 ** 10 * constants.Mpc / constants.M_sun

    @property
    def trans_dist_L(self):
        return self.cosmo.background.dist_trans_a(self.a_L)[0]

    @property
    def trans_dist_S(self):
        return self.cosmo.background.dist_trans_a(self.a_S)[0]

