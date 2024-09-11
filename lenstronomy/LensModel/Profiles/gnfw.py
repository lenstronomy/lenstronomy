__author__ = "ajshajib", "dgilman", "sibirrer"

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from scipy.interpolate import interp1d
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["GNFW"]


class GNFW(LensProfileBase):
    """
    This class computes the lensing quantities of a generalized NFW profile:

    .. math::
        \\rho(r) = \\frac{\\rho_{\\rm s}} { (r/r_{\\rm s}})^{\\gamma_{\\rm in}} * (1 + r/r_{\\rm
        s})^{3 - {\\gamma_{\\rm in}}}

    This class uses the normalization parameter `kappa_s` defined as:

    .. math::
        kappas_{\\rm s} = \\frac{\\rho_{\\rm s} r_{\\rm s}}{\\Sigma_{\\rm crit}}

    Some expressions are obtained from Keeton 2001
    https://ui.adsabs.harvard.edu/abs/2001astro.ph..2341K/abstract. See and cite the
    references therein.
    """

    model_name = "GNFW"
    _s = 0.001  # numerical limit for minimal radius
    param_names = ["Rs", "alpha_Rs", "gamma_in", "center_x", "center_y"]
    lower_limit_default = {
        "Rs": 0,
        "alpha_Rs": 0,
        "gamma_in": 0.0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "Rs": 100,
        "alpha_Rs": 10.0,
        "gamma_in": 3.0,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, trapezoidal_integration=False, integration_steps=1000):
        """

        :param trapezoidal_integrate: bool, if True, the numerical integral is performed
         with the trapezoidal rule, otherwise with ~scipy.integrate.quad
        :param integration_steps: number of steps in the trapezoidal integral
        """
        super(GNFW, self).__init__()
        self._integration_steps = integration_steps
        if trapezoidal_integration:
            self._integrate = self._trapezoidal_integrate
        else:
            self._integrate = self._quad_integrate

        self.alpha_1_interp = interp1d(_GAMMA_INS, _ALPHA_1S, kind="cubic")

    def function(self, x, y, Rs, alpha_Rs, gamma_in, center_x=0, center_y=0):
        """Potential of gNFW profile.

        :param x: angular position
        :type x: float/numpy array
        :param y: angular position
        :type y: float/numpy array
        :param Rs: angular turn over point
        :type Rs: float
        :param alpha_Rs: deflection (angular units) at projected Rs
        :type alpha_Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: potential at radius r
        :rtype: float
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        r = np.maximum(r, self._s)

        if Rs < 0.0000001:
            Rs = 0.0000001

        kappa_s = self.alpha_Rs_to_kappa_s(Rs, alpha_Rs, gamma_in)

        if isinstance(r, int) or isinstance(r, float):
            return self._num_integral_potential(r, Rs, kappa_s, gamma_in)
        else:
            # TODO: currently the numerical integral is done one by one. More efficient is sorting the radial list and
            # then perform one numerical integral reading out to the radial points
            f_ = []
            for _r in r:
                f_.append(self._num_integral_potential(_r, Rs, kappa_s, gamma_in))
            return np.array(f_)

    def _num_integral_potential(self, r, Rs, kappa_s, gamma_in):
        """Compute the numerical integral of the potential.

        :param r: radius of interest
        :type r: float
        :param Rs: scale radius
        :type Rs: float
        :param kappa_s: convergence at Rs
        :type kappa_s: float
        :type alpha_Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: potential at radius r
        :rtype: float
        """

        def _integrand(x):
            return self.alpha(x, Rs, kappa_s, gamma_in)

        return quad(_integrand, a=0, b=r)[0]

    def derivatives(self, x, y, Rs, alpha_Rs, gamma_in, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function.

        :param x: angular position
        :type x: float/numpy array
        :param y: angular position
        :type y: float/numpy array
        :param Rs: angular turn over point
        :type Rs: float
        :param alpha_Rs: deflection (angular units) at projected Rs
        :type alpha_Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: deflection angle in x, deflection angle in y
        :rtype: float, float
        """
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R, self._s)

        kappa_s = self.alpha_Rs_to_kappa_s(Rs, alpha_Rs, gamma_in)

        f_r = self.alpha(R, Rs, kappa_s, gamma_in)
        f_x = f_r * x_ / R
        f_y = f_r * y_ / R

        return f_x, f_y

    def hessian(self, x, y, Rs, alpha_Rs, gamma_in, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy.

        :param x: angular position
        :type x: float/numpy array
        :param y: angular position
        :type y: float/numpy array
        :param Rs: angular turn over point
        :type Rs: float
        :param alpha_Rs: deflection (angular units) at projected Rs
        :type alpha_Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: f_xx, f_xy, f_xy, f_yy
        :rtype: float, float, float, float
        """
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)

        kappa_s = self.alpha_Rs_to_kappa_s(Rs, alpha_Rs, gamma_in)

        kappa = self.kappa(R, Rs, kappa_s, gamma_in)
        f_r = self.alpha(R, Rs, kappa_s, gamma_in)
        f_rr = 2 * kappa - f_r / R

        cos_t = x_ / R
        sin_t = y_ / R

        f_xx = cos_t**2 * f_rr + sin_t**2 / R * f_r
        f_yy = sin_t**2 * f_rr + cos_t**2 / R * f_r
        f_xy = cos_t * sin_t * f_rr - cos_t * sin_t / R * f_r

        return f_xx, f_xy, f_xy, f_yy

    def density(self, R, Rs, rho0, gamma_in):
        """Three dimensional generalized NFW profile.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization
        :type rho0: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: rho(R) density
        :rtype: float
        """
        return rho0 * (R / Rs) ** -gamma_in * (1 + R / Rs) ** (gamma_in - 3)

    def density_lens(self, R, Rs, alpha_Rs, gamma_in):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param alpha_Rs: deflection at Rs
        :type alpha_Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: density at radius R
        :rtype: float
        """
        kappa_s = self.alpha_Rs_to_kappa_s(Rs=Rs, alpha_Rs=alpha_Rs, gamma_in=gamma_in)
        rho0 = self.kappa_s_to_rho0(kappa_s, Rs)
        return self.density(R, Rs, rho0, gamma_in)

    def density_2d(self, x, y, Rs, rho0, gamma_in, center_x=0, center_y=0):
        """Projected two dimenstional NFW profile (kappa*Sigma_crit)

        :param x: x-coordinate
        :type x: float/numpy array
        :param y: y-coordinate
        :type y: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        kappa_s = self.rho0_to_kappa_s(rho0, Rs)

        return self.kappa(R, Rs, kappa_s, gamma_in)

    def mass_3d(self, R, Rs, rho0, gamma_in):
        """Mass enclosed a 3d sphere or radius r.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: mass enclosed a 3d sphere or radius r
        :rtype: float
        """
        M_0 = 4 * np.pi * rho0 * Rs**3 / (3 - gamma_in)
        x = R / Rs
        return (
            M_0
            * x ** (3 - gamma_in)
            * hyp2f1(3 - gamma_in, 3 - gamma_in, 4 - gamma_in, -x)
        )

    def mass_3d_lens(self, R, Rs, alpha_Rs, gamma_in):
        """Mass enclosed a 3d sphere or radius r given a lens parameterization with
        angular units.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param alpha_Rs: deflection at Rs
        :type alpha_Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: mass enclosed a 3d sphere or radius r
        :rtype: float
        """
        kappa_s = self.alpha_Rs_to_kappa_s(Rs=Rs, alpha_Rs=alpha_Rs, gamma_in=gamma_in)
        rho0 = self.kappa_s_to_rho0(kappa_s, Rs)
        return self.mass_3d(R, Rs, rho0, gamma_in)

    def _trapezoidal_integrate(self, func, x, gamma_in):
        """Integrate a function using the trapezoid rule.

        :param func: function to integrate
        :type func: function
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param steps: number of steps
        :type steps: int
        :return: integral
        :rtype: float
        """
        steps = self._integration_steps
        y = np.linspace(1e-10, 1 - 1e-10, steps)
        dy = y[1] - y[0]

        weights = np.ones(steps)
        weights[0] = 0.5
        weights[-1] = 0.5

        if isinstance(x, int) or isinstance(x, float):
            integral = np.sum(func(y, x, gamma_in) * dy * weights)
        else:
            x_flat = x.flatten()
            ys = np.repeat(y[:, np.newaxis], len([x_flat]), axis=1)

            integral = np.sum(
                func(ys, x_flat, gamma_in) * dy * weights[:, np.newaxis], axis=0
            )
            integral = integral.reshape(x.shape)

        return integral

    def _quad_integrate(self, func, x, gamma_in):
        """Integrate a function using the trapezoid rule.

        :param func: function to integrate
        :type func: function
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param steps: number of steps
        :type steps: int
        :return: integral
        :rtype: float
        """
        if isinstance(x, int) or isinstance(x, float):
            integral = quad(func, a=0, b=1, args=(x, gamma_in))[0]
        else:
            integral = np.zeros_like(x)

            for i in range(len(x)):
                integral[i] = quad(func, a=0, b=1, args=(x[i], gamma_in))[0]

        return integral

    def _alpha_integrand(self, y, x, gamma_in):
        """Integrand of the deflection angel integral.

        :param y: integration variable
        :type y: np.array
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: integrand of the deflection angel integral
        """
        return (y + x) ** (gamma_in - 3) * (1 - np.sqrt(1 - y**2)) / y

    def _kappa_integrand(self, y, x, gamma_in):
        """Integrand of the deflection angel integral in eq. (57) of Keeton 2001.

        :param y: integration variable
        :type y: np.array
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: integrand of the deflection angel integral
        """
        return (y + x) ** (gamma_in - 4) * (1 - np.sqrt(1 - y**2))

    def alpha(self, R, Rs, kappa_s, gamma_in):
        """Deflection angel of gNFW profile along the radial direction.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param kappa_s: convergence at `Rs`
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: deflection angel at radius R
        :rtype: float
        """
        # R = np.maximum(R, self._s)
        x = R / Rs
        x = np.maximum(x, self._s)

        integral = self._integrate(self._alpha_integrand, x, gamma_in)

        alpha = (
            4
            * kappa_s
            * Rs
            * x ** (2 - gamma_in)
            * (
                hyp2f1(3 - gamma_in, 3 - gamma_in, 4 - gamma_in, -x) / (3 - gamma_in)
                + integral
            )
        )

        return alpha

    def kappa(self, R, Rs, kappa_s, gamma_in):
        """Convergence of gNFW profile along the radial direction.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param kappa_s: convergence at `Rs`
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: convergence at radius R
        :rtype: float
        """
        x = R / Rs
        x = np.maximum(x, self._s)

        integral = self._integrate(self._kappa_integrand, x, gamma_in)

        kappa = (
            2
            * kappa_s
            * x ** (1 - gamma_in)
            * ((1 + x) ** (gamma_in - 3) + (3 - gamma_in) * integral)
        )

        return kappa

    def kappa_s_to_alpha_Rs(self, kappa_s, Rs, gamma_in):
        """Convert the convergence at Rs to the density normalization.

        :param kappa_s: convergence at `Rs`
        :type kappa_s: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: rho0
        :rtype: float
        """
        return self.alpha(R=Rs, Rs=Rs, kappa_s=kappa_s, gamma_in=gamma_in)

    def alpha_Rs_to_kappa_s(self, Rs, alpha_Rs, gamma_in):
        """
        Convert the deflection at Rs to the convergence at Rs.

        :param Rs: scale radius
        :type Rs: float
        :param alpha_Rs: deflection at Rs
        :type alpha_Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: kappa_s
        :rtype: float
        """
        alpha_Rs_for_kappa_s_1 = self._get_alpha_Rs_for_kappa_s_1(Rs, gamma_in)
        kappa_s = alpha_Rs / alpha_Rs_for_kappa_s_1

        return kappa_s

    def rho02alpha(self, rho0, Rs, gamma_in):
        """Convenience function to compute alpha_Rs from rho0.

        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: alpha_Rs
        :rtype: float
        """
        kappa_s = self.rho0_to_kappa_s(rho0, Rs)
        return self.kappa_s_to_alpha_Rs(kappa_s, Rs, gamma_in)

    def alpha2rho0(self, alpha_Rs, Rs, gamma_in):
        """Convenience function to compute rho0 from alpha_Rs.

        :param alpha_Rs: deflection at Rs
        :type alpha_Rs: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: rho0
        :rtype: float
        """
        kappa_s = self.alpha_Rs_to_kappa_s(Rs, alpha_Rs, gamma_in)
        return self.kappa_s_to_rho0(kappa_s, Rs)

    @staticmethod
    def rho0_to_kappa_s(rho0, Rs):
        """Convenience function to compute rho0 from alpha_Rs.

        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: kappa_s
        :rtype: float
        """
        return rho0 * Rs

    @staticmethod
    def kappa_s_to_rho0(kappa_s, Rs):
        """Convenience function to compute rho0 from kappa_s. The returned rho_0 is
        normalized with $\\Sigma_{\\rm crit}$.

        :param kappa_s: convergence at `Rs`
        :type kappa_s: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: rho0
        :rtype: float
        """
        return kappa_s / Rs

    def _get_alpha_Rs_for_kappa_s_1(self, Rs, gamma_in):
        """Compute the deflection at Rs.

        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: alpha_Rs for kappa_s = 1
        :rtype: float
        """
        return Rs * self.alpha_1_interp(gamma_in)


_GAMMA_INS = np.linspace(0.0, 2.99, 300)
_ALPHA_1S = np.array(
    [
        0.56074461,
        0.56471414,
        0.56871936,
        0.57276066,
        0.57683845,
        0.58095312,
        0.58510508,
        0.58929475,
        0.59352254,
        0.59778888,
        0.6020942,
        0.60643893,
        0.61082352,
        0.61524842,
        0.61971407,
        0.62422094,
        0.6287695,
        0.63336021,
        0.63799357,
        0.64267005,
        0.64739015,
        0.65215436,
        0.6569632,
        0.66181719,
        0.66671683,
        0.67166266,
        0.67665523,
        0.68169506,
        0.68678272,
        0.69191876,
        0.69710375,
        0.70233827,
        0.7076229,
        0.71295824,
        0.71834489,
        0.72378345,
        0.72927455,
        0.73481882,
        0.74041689,
        0.74606941,
        0.75177705,
        0.75754046,
        0.76336032,
        0.76923732,
        0.77517216,
        0.78116555,
        0.7872182,
        0.79333084,
        0.79950422,
        0.80573909,
        0.81203622,
        0.81839636,
        0.82482033,
        0.83130891,
        0.83786292,
        0.84448319,
        0.85117055,
        0.85792585,
        0.86474997,
        0.87164377,
        0.87860816,
        0.88564403,
        0.89275232,
        0.89993396,
        0.9071899,
        0.91452111,
        0.92192857,
        0.92941329,
        0.93697628,
        0.94461856,
        0.95234121,
        0.96014527,
        0.96803184,
        0.97600202,
        0.98405694,
        0.99219773,
        1.00042556,
        1.00874161,
        1.01714708,
        1.02564319,
        1.03423119,
        1.04291234,
        1.05168792,
        1.06055926,
        1.06952768,
        1.07859454,
        1.08776121,
        1.09702911,
        1.10639967,
        1.11587433,
        1.1254546,
        1.13514196,
        1.14493797,
        1.15484419,
        1.16486221,
        1.17499365,
        1.18524018,
        1.19560347,
        1.20608525,
        1.21668725,
        1.22741128,
        1.23825913,
        1.24923267,
        1.26033379,
        1.27156439,
        1.28292645,
        1.29442197,
        1.30605298,
        1.31782157,
        1.32972984,
        1.34177998,
        1.35397417,
        1.36631468,
        1.3788038,
        1.39144387,
        1.40423728,
        1.41718648,
        1.43029394,
        1.44356222,
        1.45699392,
        1.47059167,
        1.48435819,
        1.49829624,
        1.51240865,
        1.52669829,
        1.54116812,
        1.55582114,
        1.57066043,
        1.58568914,
        1.60091048,
        1.61632772,
        1.63194424,
        1.64776346,
        1.6637889,
        1.68002416,
        1.69647289,
        1.71313887,
        1.73002594,
        1.74713805,
        1.7644792,
        1.78205355,
        1.79986529,
        1.81791877,
        1.83621841,
        1.85476874,
        1.87357442,
        1.89264021,
        1.91197098,
        1.93157173,
        1.9514476,
        1.97160384,
        1.99204584,
        2.01277914,
        2.03380939,
        2.05514243,
        2.07678422,
        2.09874088,
        2.12101873,
        2.1436242,
        2.16656394,
        2.18984476,
        2.21347366,
        2.23745783,
        2.26180467,
        2.28652178,
        2.31161696,
        2.33709826,
        2.36297394,
        2.3892525,
        2.4159427,
        2.44305354,
        2.47059429,
        2.49857452,
        2.52700405,
        2.55589301,
        2.58525186,
        2.61509136,
        2.6454226,
        2.67625703,
        2.70760647,
        2.7394831,
        2.77189949,
        2.80486862,
        2.83840392,
        2.87251922,
        2.90722884,
        2.94254757,
        2.97849071,
        3.01507408,
        3.05231404,
        3.09022753,
        3.12883208,
        3.16814586,
        3.20818765,
        3.24897697,
        3.29053401,
        3.33287972,
        3.37603585,
        3.42002494,
        3.46487041,
        3.51059658,
        3.55722873,
        3.6047931,
        3.65331701,
        3.70282886,
        3.75335822,
        3.80493586,
        3.85759383,
        3.91136555,
        3.96628584,
        4.02239102,
        4.07971902,
        4.1383094,
        4.19820351,
        4.25944456,
        4.32207775,
        4.38615034,
        4.45171183,
        4.51881407,
        4.58751138,
        4.65786074,
        4.72992195,
        4.80375778,
        4.87943418,
        4.95702052,
        5.03658974,
        5.11821867,
        5.20198822,
        5.28798372,
        5.3762952,
        5.46701773,
        5.56025176,
        5.65610357,
        5.75468562,
        5.85611709,
        5.96052435,
        6.06804152,
        6.17881111,
        6.29298466,
        6.41072347,
        6.53219947,
        6.65759603,
        6.78710899,
        6.92094777,
        7.05933652,
        7.20251547,
        7.35074245,
        7.50429449,
        7.66346971,
        7.82858933,
        8.0,
        8.17807639,
        8.36322408,
        8.55588285,
        8.75653039,
        8.96568649,
        9.18391782,
        9.41184336,
        9.65014063,
        9.89955279,
        10.16089686,
        10.43507314,
        10.72307617,
        11.02600745,
        11.34509024,
        11.681687,
        12.03731975,
        12.41369431,
        12.81272892,
        13.23658849,
        13.68772572,
        14.16893062,
        14.68339078,
        15.23476486,
        15.82727325,
        16.46581027,
        17.15608456,
        17.90479608,
        18.71986129,
        19.61070275,
        20.58862559,
        21.66731279,
        22.86348565,
        24.1977973,
        25.69606125,
        27.39097143,
        29.32455932,
        31.55178513,
        34.14592464,
        37.20689417,
        40.87457069,
        45.35099179,
        50.9392058,
        58.11542188,
        67.67331019,
        81.04146222,
        101.07704164,
        134.44673674,
        201.15056993,
        401.18856449,
    ]
)
