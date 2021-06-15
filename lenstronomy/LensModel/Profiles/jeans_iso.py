__author__ = 'jhodonnell'

import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import root

import warnings

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['SIDMJeans']

# ## Analytic Solution
#
# As mentioned above, for x close to zero we have an exact solution. Below a
# small threshold, we will approx y(x)/y'(x) with these solutions, above that
# threshold we will use odeint.


def _y_approx(x):
    return np.log(1 - np.tanh(x/np.sqrt(6))**2)


def _dy_approx(x):
    return -np.sqrt(2/3)*np.tanh(x/np.sqrt(6))


def _ddy_approx(x):
    return -1/(3*np.cosh(x/np.sqrt(6)))


class _JeansSolutionInterp:
    '''
    Solves the Jeans method differential equation, and boundary conditions
    between the isothermal core and NFW exterior.

    Closely follows Robertson 2021:
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.4610R/abstract

    The density of an isothermal gas under gravity follows the Jeans equation,
    given by eqn. 8 in Robertson 2021

    .. math::
        \\sigma_0^2 \\nabla^2 \\log \\rho = - 4 \\pi G \\rho

    Here we assume only dark matter, but baryonic contributions can be added to
    \\rho as well.

    Robertson 2021 use the following change of variables:

    .. math::
        y = \\log(\\rho/\\rho_0)
        r_0^2 = \\sigma_0^2/4\\pi G \\rho_0
        x = r/r_0

    And the equation becomes:

    .. math:
        d^2y/dx^2 + (2/x) (dy/dx) + exp(y) = 0

    The two boundary conditions are that:

    1) The density is continuous between the isothermal core and NFW exterior
    2) The total mass contained in the isothermal core is the same as it would
    be for an NFW
    '''

    def _solve_jeans_function(self):
        '''
        Solve the isothermal Jeans differential eqn
        '''
        def _jeans_function_integrand(y, x):
            y, dy = y
            ddy = -(np.exp(y) + 2*dy / x)
            return np.array([dy, ddy])

        xspan = np.logspace(-3, 2, 10000)
        start_x = (xspan[0])
        solution = integrate.odeint(
            _jeans_function_integrand,
            y0=[_y_approx(start_x), _dy_approx(start_x)],
            t=xspan
        )

        self._y_start_x = start_x
        self._xspan_y = xspan
        self._y_interp = interp1d(
            xspan, solution[:, 0], fill_value='extrapolate'
        )
        self._dy_interp = interp1d(
            xspan, solution[:, 1], fill_value='extrapolate'
        )

    def _solve_mass(self):
        '''
        Compute the enclosed mass of the isothermal Jeans eqn
        '''
        def mass_integrand(y_, x):
            return x*x*np.exp(self.y(x))

        xs_mass = np.concatenate([[0.0], np.logspace(-3, 2, 9999)])
        mass_integrated = integrate.odeint(mass_integrand, y0=[0], t=xs_mass).flatten()

        self._xspan_mass = xs_mass
        self._mass_interp = interp1d(
            xs_mass, mass_integrated, fill_value='extrapolate'
        )

    def y(self, x):
        '''
        Give the (unitless) solution to the reduced Jeans function y(x)
        '''
        if not hasattr(self, '_y_interp'):
            self._solve_jeans_function()

        if isinstance(x, (int, float)):
            if x < self._y_start_x:
                return _y_approx(x)
            else:
                return self._y_interp(x)

        output = np.empty_like(x)
        output[x <= self._y_start_x] = _y_approx(x[x <= self._y_start_x])
        output[x > self._y_start_x] = self._y_interp(x[x > self._y_start_x])
        return output

    def enclosed_mass_3d(self, x):
        '''
        Give the (unitless) mass enclosed within the sphere r < r0*x
        '''
        if not hasattr(self, '_mass_interp'):
            self._solve_mass()

        if isinstance(x, (int, float)):
            if x < self._xspan_mass.min():
                return 0.0
            elif x > self._xspan_mass.max():
                xmax = self._xspan_mass.max()
                return (
                    self._mass_interp(xmax)*(x/xmax)**2
                )
            return self._mass_interp(x)

        xmin, xmax = self._xspan_mass.min(), self._xspan_mass.max()
        output = np.empty_like(x)
        output[x < xmin] = 0
        output[(x >= xmin) & (x <= xmax)] = self._mass_interp(
            x[(x >= xmin) & (x <= xmax)]
        )
        output[x > xmax] = self._mass_interp(xmax)*(x[x > xmax]/xmax)**2
        return output

    @staticmethod
    def solve_nfw(a):
        '''
        Inverts the unitless NFW profile:

            f(x) = 1 / (x (1 + x)^2)

        I.e., gives x such that f(x) = a.
        '''
        term = 27*a**2 + 2*a**3 - 3*a**2*np.sqrt(3*(27 + 4*a))
        return (
            -4
            + 2*a/((term/2)**(1/3))
            + 2**(2/3)*term**(1/3) / a
        ) / 6

    def _solve_boundary(self, a, guess=None):
        if guess is None:
            x0 = [0.0, 0.0]
        else:
            x0 = np.log(guess)

        def _unitless_jeans_residual(x):
            log_b, log_c = x
            b, c = np.exp([log_b, log_c])

            # Enclosed mass
            exp_mass = -a / (1 + a) + np.log(1 + a)
            actual_mass = c * b**3 * self.enclosed_mass_3d(a/b)

            # Density
            exp_density = 1/(a * (1 + a)**2)
            actual_density = c * np.exp(self.y(a / b))

            return 10*np.log([
                exp_mass / actual_mass,
                exp_density / actual_density
            ])

        # Solve!
        soln = root(_unitless_jeans_residual,
                    method='lm',
                    x0=x0,
                    tol=1e-14,
                    options=dict(ftol=1e-14))

        # Check if failure
        if not soln.success:
            print(soln)
            raise ValueError(f'Jeans solution couldn\'t converge for a = {a}')

        b, c = np.exp(soln.x)
        x = a / b

        # Warn if outside of interpolation range
        if x > self._xspan_mass[-1]:
            warnings.warn(
                f'Jeans solution (b, c) = ({b:.2e}, {c:.2e})'
                f' for a = {a:.2e} is out of bounds of mass interpolation'
            )
        if x > self._xspan_y[-1]:
            warnings.warn(
                f'Jeans solution (b, c) = ({b:.2e}, {c:.2e})'
                f' for a = {a:.2e} is out of bounds of Jeans equation solution'
                ' interpolation'
            )

        return b, c

    def _make_boundary_interp(self):
        # Solve boundary conditions on a large grid of a values
        # That way - we will have high-quality initial guesses for (b, c)
        a_range = np.logspace(-2, 1, 1000)
        solved_b_c = []
        guess = None
        for a in a_range:
            try:
                solved_b_c.append(self._solve_boundary(a, guess=guess))
                guess = solved_b_c[-1]
            except ValueError:
                print('jeans solving failed at a =', a)
                solved_b_c.append([0.0, 0.0])

        solved_b_c = np.array(solved_b_c)

        self._interp_log_b_guess = interp1d(
            a_range, np.log(solved_b_c[:, 0]), fill_value='extrapolate'
        )
        self._interp_log_c_guess = interp1d(
            a_range, np.log(solved_b_c[:, 1]), fill_value='extrapolate'
        )

    def solve_jeans_boundary(self, a):
        '''
        Solve the boundary conditions between the isothermal core and the
        exterior NFW profile.

        :param a: The ratio of the transition radius to NFW scale radius.
                  a = r1 / rs

        :returns: 2-tuple of (b, c)
                  b = r0 / rs, the Jeans radius in terms of the NFW scale
                  c = rho_iso / rho_NFW, the Jeans density in terms of NFW density
        '''
        if not hasattr(self, '_interp_log_b_guess'):
            self._make_boundary_interp()

        log_b_guess = self._interp_log_b_guess(a)
        log_c_guess = self._interp_log_c_guess(a)

        guess = np.exp([log_b_guess, log_c_guess])
        b, c = self._solve_boundary(a, guess=guess)

        return b, c

    def unitless_jeans_profile(self, xs, a, b, c):
        '''
        Computes the Jeans method profile: NFW for x > a,
        isothermal jeans for x < a.
        '''
        if isinstance(xs, (int, float)):
            if xs >= a:
                return 1/(xs * (1 + xs)**2)
            return c*np.exp(self.y(xs/b))

        # xs are r/rs
        output = np.empty_like(xs)
        output[xs >= a] = 1/(xs[xs >= a] * (1 + xs[xs >= a])**2)
        output[xs < a] = (
            c*np.exp(self.y(xs[xs < a]/b))
        )
        return output


# Global jeans soln interp
_jeans_soln = _JeansSolutionInterp()


class SIDMJeans(LensProfileBase):
    '''
    Lensing profile for isothermal Jeans modeling of self-interacting dark matter halos,
    as explained thoroughly in Robertson 2021: https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.4610R/abstract

    Use the function :meth:`lenstronomy.Cosmo.lens_cosmo.LensCosmo.jeans_physical2angle` to
    convert physical quantities to the lensing parameters.
    '''

    # Rs = NFW Scale radius
    # alpha = r_1 / Rs, what's called a in the class above
    # beta = r_0 / Rs, what's called b in the class above
    # gamma = rho_0 / rho_NFW, what's called c in the class above
    param_names = ['Rs', 'alpha', 'beta', 'gamma', 'rho0', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha': 0, 'beta': 0, 'gamma': 0, 'rho0': 0,
                           'center_x': -100, 'center_y': -100}
    lower_limit_default = {'Rs': 100, 'alpha': 0, 'beta': 1000, 'gamma': 100, 'rho0': 1000,
                           'center_x': -100, 'center_y': -100}

    def _convergence_nfw(self, radius, Rs, rho0):
        '''
        The analytical convergence for an NFW halo.

        Gives the projected NFW integral at r = r/rs.
        '''
        x = radius / Rs

        kappas = np.empty_like(x)
        xlow = x[x < 1]
        kappas[x < 1] = (
            2/(xlow**2 - 1) * (1 - 2*np.arctanh(np.sqrt((1 - xlow)/(1 + xlow))) / np.sqrt(1 - xlow**2))
        )
        kappas[x == 1] = 1/3
        xhigh = x[x > 1]
        kappas[x > 1] = (
            2/(xhigh**2 - 1) * (1 - 2*np.arctan(np.sqrt((xhigh - 1)/(1 + xhigh)))/np.sqrt(xhigh**2 - 1))
        )

        return rho0 * Rs * kappas

    def _convergence_integral(self, radius, Rs, alpha, beta, gamma, rho0):
        '''
        Numerically integrates the projection of the isothermal/NFW Jeans profile.
        '''
        def integrand(x, B):
            rprime = np.sqrt(x**2 + B**2)
            return _jeans_soln.unitless_jeans_profile(rprime, alpha, beta, gamma)

        xs = radius / Rs

        kappas = np.empty_like(radius)
        for i, x in enumerate(xs):
            kappas[i] = 2*integrate.quad(integrand, 0, 100, args=(x,))[0]

        return rho0 * Rs * kappas

    def _convergence_sidm(self, radius, Rs, alpha, beta, gamma, rho0):
        '''
        Combined convergence for the halo
        '''
        is_scalar = isinstance(radius, (int, float))
        radius = np.array(radius)

        x = radius / Rs

        kappas = np.empty_like(radius)
        kappas[x >= alpha] = self._convergence_nfw(
            radius[x >= alpha], Rs, rho0
        )
        kappas[x < alpha] = self._convergence_integral(
            radius[x < alpha], Rs, alpha, beta, gamma, rho0
        )

        if is_scalar:
            return kappas[()]
        return kappas

    def _potential(self, thetas, Rs, alpha, beta, gamma, rho0):
        '''
        Compute the lensing potential at angles `thetas` from the halo
        '''
        inner = integrate.odeint(
            lambda _, theta1: theta1 * self._convergence_sidm(
                theta1, Rs, alpha, beta, gamma, rho0
            ),
            # Dummy y0
            [0.0],
            np.concatenate(([0.0], thetas))
        ).flatten()[1:] * 2 * np.log(thetas)

        # Negative because we integrate backwards
        outer = -integrate.odeint(
            # The integration algorithm likes to extend past the bounds of integration,
            # so we use abs(theta1) to allow it to explore past theta = 0 without
            # blowing up
            lambda _, theta1: np.abs(theta1) * np.log(np.abs(theta1)) * self._convergence_sidm(
                np.abs(theta1), Rs, alpha, beta, gamma, rho0
            ),
            # Dummy y0
            [0.0],
            # Start from 100 and integrate BACKWARD
            np.concatenate(([100.0], thetas[::-1]))
        # Flip backwards again
        ).flatten()[1:][::-1] * 2

        return inner + outer

    def _deflection(self, thetas, Rs, alpha, beta, gamma, rho0):
        output = np.empty_like(thetas)

        near_zero = thetas < 1e-2
        output[near_zero] = (
            2 * thetas[near_zero]
            * self._convergence_sidm(
                thetas[near_zero], Rs, alpha, beta, gamma, rho0)
        )

        larger = thetas >= 1e-2
        output[larger] = integrate.odeint(
            lambda _y, theta1: theta1*self._convergence_sidm(
                theta1, Rs, alpha, beta, gamma, rho0
            ),
            # dummy y0 - value unused
            [0.0],
            np.concatenate(([0.0], thetas[larger]))
        ).flatten()[1:] * 2 / thetas[larger]

        return output

    def function(self, x, y, Rs, alpha, beta, gamma, rho0, center_x=0, center_y=0):
        """
        lensing potential

        :param x: x-coordinate, arcsec
        :param y: y-coordinate, arcsec
        :param Rs: NFW scale radius, arcsec
        :param alpha: isothermal/NFW transition radius relative to Rs
        :param beta: isothermal scale radius r0 relative to Rs
        :param gamma: core density relative to rho0
        :param rho0: NFW density, angular units

        :return: the lensing potential at x and y
        """
        _x = x - center_x
        _y = y - center_y

        thetas = np.sqrt(_x**2 + _y**2)

        potential = np.empty_like(thetas)
        pot_shape = potential.shape
        sort_order = np.argsort(thetas, axis=None)

        # Sort them by distance first
        # TODO: optimizations? interpolate?
        potential = potential.flatten()
        potential[sort_order] = self._potential(
            thetas[sort_order], Rs, alpha, beta, gamma, rho0
        )

        return potential.reshape(pot_shape)

    def derivatives(self, x, y, Rs, alpha, beta, gamma, rho0, center_x=0, center_y=0):
        """
        deflection angles

        :param x: x-coordinate, arcsec
        :param y: y-coordinate, arcsec
        :param Rs: NFW scale radius, arcsec
        :param alpha: isothermal/NFW transition radius relative to Rs
        :param beta: isothermal scale radius r0 relative to Rs
        :param gamma: core density relative to rho0
        :param rho0: NFW density, angular units

        :return: x-deflections, y-deflections
        """
        _x = x - center_x
        _y = y - center_y

        thetas = np.sqrt(_x**2 + _y**2)

        deflection = np.empty_like(thetas)
        pot_shape = deflection.shape
        sort_order = np.argsort(thetas, axis=None)

        # Sort them by distance first
        # TODO: optimizations? interpolate?
        deflection = deflection.flatten()
        deflection[sort_order] = self._deflection(
            thetas[sort_order], Rs, alpha, beta, gamma, rho0
        )
        deflection = deflection.reshape(pot_shape)

        f_x = (_x/thetas) * deflection
        f_y = (_y/thetas) * deflection

        # compute f_x, f_y and return
        return f_x, f_y

    def hessian(self, *args, **kwargs):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2

        :param x: x-coordinate, arcsec
        :param y: y-coordinate, arcsec
        :param Rs: NFW scale radius, arcsec
        :param alpha: isothermal/NFW transition radius relative to Rs
        :param beta: isothermal scale radius r0 relative to Rs
        :param gamma: core density relative to rho0
        :param rho0: NFW density, angular units

        :return: raise as definition is not defined
        """
        raise NotImplementedError

    def density_lens(self, radius, Rs, alpha, beta, gamma, rho0):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param radius: radii at which to compute density, numpy array or scalar
        :param Rs: NFW scale radius, arcsec
        :param alpha: isothermal/NFW transition radius relative to Rs
        :param beta: isothermal scale radius r0 relative to Rs
        :param gamma: core density relative to rho0
        :param rho0: NFW density, angular units

        :return: 3D density in angular units, same shape as radius
        """
        x = radius / Rs
        unitless_density = _jeans_soln.unitless_jeans_profile(
            x, alpha, beta, gamma
        )
        return unitless_density * rho0

    def density_2d(self, x, y, Rs, alpha, beta, gamma, rho0, center_x=0, center_y=0):
        """
        Convergence at a grid of (x, y)

        :param x: x-coordinate, arcsec
        :param y: y-coordinate, arcsec
        :param Rs: NFW scale radius, arcsec
        :param alpha: isothermal/NFW transition radius relative to Rs
        :param beta: isothermal scale radius r0 relative to Rs
        :param gamma: core density relative to rho0
        :param rho0: NFW density, angular units

        :return: convergence, numpy array or scalar of same shape as x&y
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        projection = self._convergence_sidm(R, Rs, alpha, beta, gamma, rho0)
        return projection

    def mass_3d_lens(self, radius, Rs, alpha, beta, gamma, rho0):
        """
        mass enclosed a 3d sphere or radius r given a lens parameterization with angular units
        For this profile those are identical.

        :param radius: radii at which to compute mass, angular units
        :param Rs: NFW scale radius, arcsec
        :param alpha: isothermal/NFW transition radius relative to Rs
        :param beta: isothermal scale radius r0 relative to Rs
        :param gamma: core density relative to rho0
        :param rho0: NFW density, angular units

        :return: raise as definition is not defined
        """
        x = radius / Rs
        if x < alpha:
            unitless_density = _jeans_soln.enclosed_mass_3d(x / beta)
        else:
            # TODO
            raise NotImplementedError

        return rho0 * Rs**3 * unitless_density

    def set_static(self, **kwargs):
        """
        pre-computes certain computations that do only relate to the lens model parameters and not to the specific
        position where to evaluate the lens model

        :param kwargs: lens model parameters
        :return: no return, for certain lens model some private self variables are initiated
        """
        pass

    def set_dynamic(self):
        """

        :return: no return, deletes pre-computed variables for certain lens models
        """
        pass
