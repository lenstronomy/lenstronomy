import numpy as np

__all__ = ['NFWParam']


class NFWParam(object):
    """
    class which contains a halo model parameters dependent on cosmology for NFW profile
    All distances are given in physical units. Mass definitions are relative to 200 crit including redshift evolution.
    The redshift evolution is cosmology dependent (dark energy).
    The H0 dependence is propagated into the input and return units.
    """

    rhoc = 2.77536627e11  # critical density [h^2 M_sun Mpc^-3]

    def __init__(self, cosmo=None):
        """

        :param cosmo: astropy.cosmology instance
        """
        from astropy.cosmology import default_cosmology

        if cosmo is None:
            cosmo = default_cosmology.get()
        self.cosmo = cosmo

    def rhoc_z(self, z):
        """

        :param z: redshift
        :return: critical density of the universe at redshift z in physical units [h^2 M_sun Mpc^-3]
        """
        return self.rhoc * (self.cosmo.efunc(z)) ** 2
        #return self.rhoc*(1+z)**3

    def M200(self, rs, rho0, c):
        """
        M(R_200) calculation for NFW profile

        :param rs: scale radius
        :type rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param c: concentration
        :type c: float [4,40]
        :return: M(R_200) density
        """
        return 4 * np.pi * rho0 * rs ** 3 * (np.log(1. + c) - c / (1. + c))

    def r200_M(self, M, z):
        """
        computes the radius R_200 crit of a halo of mass M in physical distances M/h

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :param z: redshift
        :type z: float
        :return: radius R_200 in physical Mpc/h
        """
        return (3*M/(4*np.pi*self.rhoc_z(z)*200))**(1./3.)

    def M_r200(self, r200, z):
        """

        :param r200: r200 in physical Mpc/h
        :param z: redshift
        :return: M200 in M_sun/h
        """
        return self.rhoc_z(z)*200 * r200**3 * 4*np.pi/3.

    def rho0_c(self, c, z):
        """
        computes density normalization as a function of concentration parameter

        :param c: concentration
        :param z: redshift
        :return: density normalization in h^2/Mpc^3 (physical)
        """
        return 200./3*self.rhoc_z(z)*c**3/(np.log(1.+c)-c/(1.+c))

    def c_rho0(self, rho0, z):
        """
        computes the concentration given density normalization rho_0 in h^2/Mpc^3 (physical) (inverse of function rho0_c)
        :param rho0: density normalization in h^2/Mpc^3 (physical)
        :param z: redshift
        :return: concentration parameter c
        """
        if not hasattr(self, '_c_rho0_interp'):
            c_array = np.linspace(0.1, 10, 100)
            rho0_array = self.rho0_c(c_array, z)
            from scipy import interpolate
            self._c_rho0_interp = interpolate.InterpolatedUnivariateSpline(rho0_array, c_array, w=None, bbox=[None, None], k=3)
        return self._c_rho0_interp(rho0)

    def c_M_z(self, M, z):
        """
        fitting function of http://moriond.in2p3.fr/J08/proceedings/duffy.pdf for the mass and redshift dependence of
        the concentration parameter

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :param z: redshift
        :type z: float >0
        :return: concentration parameter as float
        """
        # fitted parameter values
        A = 5.22
        B = -0.072
        C = -0.42
        M_pivot = 2.*10**12
        return A*(M/M_pivot)**B*(1+z)**C

    def nfw_Mz(self, M, z):
        """
        returns all needed parameter (in physical units modulo h) to draw the profile of the main halo
        r200 in physical Mpc/h
        rho_s in  h^2/Mpc^3 (physical)
        Rs in Mpc/h physical
        c unit less
        """
        c = self.c_M_z(M, z)
        r200 = self.r200_M(M, z)
        rho0 = self.rho0_c(c, z)
        Rs = r200/c
        return r200, rho0, c, Rs
