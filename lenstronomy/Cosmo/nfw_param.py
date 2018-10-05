import numpy as np


class NFWParam(object):
    """
    class which contains a halo model parameters dependent on cosmology for NFW profile
    all distances are given in comoving coordinates
    """

    rhoc = 2.77536627e11  # critical density [h^2 M_sun Mpc^-3]

    def rhoc_z(self, z):
        """

        :param z: redshift
        :return: scaled critical density as a function of redshift (attention, this is not rho_crit(z))
        """
        return self.rhoc*(1+z)**3

    def M200(self, Rs, rho0, c):
        """
        M(R_200) calculation for NFW profile

        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param c: concentration
        :type c: float [4,40]
        :return: M(R_200) density
        """
        return 4*np.pi*rho0*Rs**3*(np.log(1.+c)-c/(1.+c))

    def r200_M(self, M):
        """
        computes the radius R_200 of a halo of mass M in comoving distances M/h

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :return: radius R_200 in comoving Mpc/h
        """
        return (3*M/(4*np.pi*self.rhoc*200))**(1./3.)

    def M_r200(self, r200):
        """

        :param r200: r200 in comoving Mpc/h
        :return: M200 in M_sol/h
        """
        return self.rhoc*200 * r200**3 * 4*np.pi/3.

    def rho0_c(self, c):
        """
        computes density normalization as a function of concentration parameter
        :return: density normalization in h^2/Mpc^3 (comoving)
        """
        return 200./3*self.rhoc*c**3/(np.log(1.+c)-c/(1.+c))

    def c_rho0(self, rho0):
        """
        computes the concentration given a comoving overdensity rho0 (inverse of function rho0_c)
        :param rho0: density normalization in h^2/Mpc^3 (comoving)
        :return: concentration parameter c
        """
        if not hasattr(self, '_c_rho0_interp'):
            c_array = np.linspace(0.1, 10, 100)
            rho0_array = self.rho0_c(c_array)
            from scipy import interpolate
            self._c_rho0_interp = interpolate.InterpolatedUnivariateSpline(rho0_array, c_array, w=None, bbox=[None, None], k=3)
        return self._c_rho0_interp(rho0)

    def c_M_z(self, M, z):
        """
        fitting function of http://moriond.in2p3.fr/J08/proceedings/duffy.pdf for the mass and redshift dependence of the concentration parameter

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

    def profileMain(self, M, z):
        """
        returns all needed parameter (in comoving units modulo h) to draw the profile of the main halo
        r200 in co-moving Mpc/h
        rho_s in  h^2/Mpc^3 (co-moving)
        Rs in Mpc/h co-moving
        c unit less
        """
        c = self.c_M_z(M, z)
        r200 = self.r200_M(M)
        rho0 = self.rho0_c(c)
        Rs = r200/c
        return r200, rho0, c, Rs