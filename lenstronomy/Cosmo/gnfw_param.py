import numpy as np
from scipy.special import hyp2f1
from scipy import interpolate
from astropy.cosmology import default_cosmology

from .nfw_param import NFWParam

__all__ = ["GNFWParam"]


class GNFWParam(object):
    """Class which contains a halo model parameters dependent on cosmology for gNFW
    profile. All distances are given in physical units.

    Mass definitions are relative to 200 crit including redshift evolution. The redshift
    evolution is cosmology dependent (dark energy). The H0 dependence is propagated into
    the input and return units.
    """

    rhoc = 2.77536627e11  # critical density [h^2 M_sun Mpc^-3]

    def __init__(self, cosmo=None):
        """

        :param cosmo: astropy.cosmology instance
        :type cosmo: astropy.cosmology instance
        """
        if cosmo is None:
            cosmo = default_cosmology.get()
        self.cosmo = cosmo
        self.nfw_param = NFWParam(cosmo)

    def rhoc_z(self, z):
        """Compute the critical density of the universe at redshift z in physical units
        [h^2 M_sun Mpc^-3].

        :param z: redshift
        :type z: float
        :return: critical density of the universe at redshift z in physical units [h^2
            M_sun Mpc^-3]
        :rtype: float
        """
        return self.nfw_param.rhoc_z(z)
        # return self.rhoc*(1+z)**3

    @staticmethod
    def M200(rs, rho0, c, gamma_in):
        """Calculation of the mass enclosed r_200 for gNFW profile defined as.

        .. math::
            M_{200} = 4 \\pi \\rho_0^{3} r_{\\rm s}^{3} \frac{c^{3 - \\gamma_{\\rm in}}} {3 - \\gamma_{\\rm in}}  {}_2F_1(3 - \\gamma_{\\rm in}, 3 - \\gamma_{\\rm in}; 4 - \\gamma_{\\rm in}; -c)

        :param rs: scale radius
        :type rs: float
        :param rho0: density normalization (characteristic density) in units mass/[distance unit of rs]^3
        :type rho0: float
        :param c: concentration
        :type c: float [4,40]
        :param gamma_in: inner slope of the gNFW profile
        :type gamma_in: float
        :return: M(R_200) mass in units of rho0 * rs^3
        :rtype: float
        """
        return (
            4
            * np.pi
            * rho0
            * rs**3
            * c ** (3 - gamma_in)
            / (3 - gamma_in)
            * hyp2f1(3 - gamma_in, 3 - gamma_in, 4 - gamma_in, -c)
        )

    def r200_M(self, M, z):
        """Compute the radius R_200 crit of a halo of mass M in physical mass M/h.

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :param z: redshift
        :type z: float
        :return: radius R_200 in physical Mpc/h
        :rtype: float or numpy array
        """
        return self.nfw_param.r200_M(M, z)

    def M_r200(self, r200, z):
        """Compute the mass M_200 of a halo of radius r_200 in physical Mpc/h.

        :param r200: r200 in physical Mpc/h
        :type r200: float
        :param z: redshift
        :type z: float
        :return: M200 in M_sun/h
        :rtype: float
        """
        return self.nfw_param.M_r200(r200, z)

    def rho0_c(self, c, z, gamma_in):
        """Computes density normalization as a function of concentration parameter.

        :param c: concentration
        :type c: float
        :param z: redshift
        :type z: float
        :param gamma_in: inner slope of the gNFW profile
        :type gamma_in: float
        :return: density normalization in h^2/Mpc^3 (physical)
        :rtype: float
        """
        return (
            200.0
            / 3
            * self.rhoc_z(z)
            * (3 - gamma_in)
            * c**gamma_in
            / hyp2f1(3 - gamma_in, 3 - gamma_in, 4 - gamma_in, -c)
        )

    def c_rho0(self, rho0, z, gamma_in):
        """Computes the concentration given density normalization rho_0 in h^2/Mpc^3
        (physical) (inverse of function rho0_c)

        :param rho0: density normalization in h^2/Mpc^3 (physical)
        :type rho0: float
        :param z: redshift
        :type z: float
        :param gamma_in: inner slope of the gNFW profile
        :type gamma_in: float
        :return: concentration parameter c
        :rtype: float
        """
        c_array = np.linspace(0.1, 30, 100)
        if not hasattr(self, "_rho0_c_gamma_in_interps"):
            gamma_in_array = np.linspace(0.1, 2.99, 100)
            self._rho0_c_gamma_in_interps = []

            for i in range(len(c_array)):
                rho0_array = self.rho0_c(c_array[i], z, gamma_in_array)

                self._rho0_c_gamma_in_interps.append(
                    interpolate.InterpolatedUnivariateSpline(
                        gamma_in_array, rho0_array, w=None, bbox=[None, None], k=3
                    )
                )

        rho0_interp = [interp(gamma_in) for interp in self._rho0_c_gamma_in_interps]

        c_rho0_interp = interpolate.InterpolatedUnivariateSpline(
            rho0_interp, c_array, w=None, bbox=[None, None], k=3
        )

        return c_rho0_interp(rho0)

    def c_M_z(self, M, z):
        """
        Fitting function of http://moriond.in2p3.fr/J08/proceedings/duffy.pdf for the mass and redshift dependence of
        the concentration parameter. Here, assuming the NFW M-c relation for the gNFW profile.

        :param M: halo mass in M_sun/h
        :type M: float or numpy array
        :param z: redshift
        :type z: float >0
        :return: concentration parameter as float
        :rtype: float
        """
        return self.nfw_param.c_M_z(M, z)

    def gnfw_Mz(self, M, z, gamma_in):
        """Returns all needed parameter (in physical units modulo h) to draw the profile
        of the main halo r200 in physical Mpc/h rho_s in  h^2/Mpc^3 (physical) Rs in
        Mpc/h physical c unit less.

        :param M: Mass in physical M_sun/h
        :type M: float
        :param z: redshift
        :type z: float
        :param gamma_in: inner slope of the gNFW profile
        :type gamma_in: float
        :return: r200, rho0, c, Rs
        :rtype: float, float, float, float
        """
        c = self.c_M_z(M, z)
        r200 = self.r200_M(M, z)
        rho0 = self.rho0_c(c, z, gamma_in)
        Rs = r200 / c
        return r200, rho0, c, Rs
