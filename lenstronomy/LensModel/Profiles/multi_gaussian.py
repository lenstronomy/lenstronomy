__all__ = ["MultiGaussian", "MultiGaussianEllipsePotential"]

import numpy as np
from lenstronomy.LensModel.Profiles.gaussian import Gaussian
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.gaussian_ellipse_potential import (
    GaussianEllipsePotential,
)


class MultiGaussian(LensProfileBase):
    """This class implements a sum of multiple circular Gaussian profiles for use in
    gravitational lensing.

    Each component in the sum is a circular Gaussian potential profile defined
    by its amplitude (`amp`) and standard deviation (`sigma`). The Gaussian
    profiles are centered at (`center_x`, `center_y`) and are scaled by an
    optional global factor (`scale_factor`).

    This model can approximate more complex smooth mass distributions by
    combining multiple Gaussians with different widths and amplitudes.
    """

    param_names = ["amp", "sigma", "center_x", "center_y", "scale_factor"]
    lower_limit_default = {
        "amp": 0,
        "sigma": 0,
        "center_x": -100,
        "center_y": -100,
        "scale_factor": 0,
    }
    upper_limit_default = {
        "amp": 100,
        "sigma": 100,
        "center_x": 100,
        "center_y": 100,
        "scale_factor": 10000,
    }

    def __init__(self):
        self.gaussian = Gaussian()
        super(MultiGaussian, self).__init__()

    def function(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """Returns the summed Gaussian potential evaluated at coordinates (x, y).

        Each component is a circular 2D Gaussian profile centered at (center_x,
        center_y), with its own amplitude and width (sigma). The total potential is the
        sum of all such Gaussian components. A scale factor is optional.

        :param x: x-coordinate(s) of the evaluation grid (array-like)
        :param y: y-coordinate(s) of the evaluation grid (array-like)
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each Gaussian component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: total potential evaluated at (x, y)
        """
        # Initialize potential
        f_ = np.zeros_like(x, dtype=float)
        # Loop through each Gaussian component
        for i in range(len(amp)):
            # Add the ith Gaussian (with scaled amplitude) to the total
            f_ += self.gaussian.function(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                center_x=center_x,
                center_y=center_y,
            )
        return f_

    def derivatives(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """Returns the gradient in both angular directions of the summed Gaussian
        potential evaluated at (x, y).

        .. math::
            \\frac{df}{dx}, \\frac{df}{dy}

        :param x: x-coordinate(s) where the gradient is evaluated
        :param y: y-coordinate(s) where the gradient is evaluated
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: :math:`\\frac{df}{dx}, \\frac{df}{dy}` of the same shape as x and y
        """
        # Initialize gradients
        f_x, f_y = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
        # Loop through each Gaussian component
        for i in range(len(amp)):
            # Compute x and y derivatives of the ith Gaussian
            f_x_i, f_y_i = self.gaussian.derivatives(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                center_x=center_x,
                center_y=center_y,
            )
            f_x += f_x_i
            f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """Returns the second derivatives of the summed Gaussian potential evaluated at
        (x, y).

        :param x: x-coordinate(s) where the gradient is evaluated
        :param y: y-coordinate(s) where the gradient is evaluated
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: :math:`\\frac{df}{dx}, \\frac{df}{dy}` of the same shape as x and y
        """
        # Initialize hessian
        f_xx, f_yy, f_xy = (
            np.zeros_like(x, dtype=float),
            np.zeros_like(x, dtype=float),
            np.zeros_like(x, dtype=float),
        )
        # Loop through each Gaussian component
        for i in range(len(amp)):
            # Get second derivatives of the ith Gaussian
            f_xx_i, f_xy_i, _, f_yy_i = self.gaussian.hessian(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                center_x=center_x,
                center_y=center_y,
            )
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_xy, f_xy, f_yy

    def density(self, r, amp, sigma, scale_factor=1):
        """Returns the 3D density profile evaluated at radius `r` for a sum of Gaussian
        components.

        :param r: radial coordinate to evaluate the density
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param scale_factor: global factor applied to each amplitude
        :return: total 3D density evaluated at `r`
        """
        # Initialize total density
        d_ = np.zeros_like(r, dtype=float)
        # Loop through each Gaussian component
        for i in range(len(amp)):
            # Add the ith Gaussian density to the total
            d_ += self.gaussian.density(r, scale_factor * amp[i], sigma[i])
        return d_

    def density_2d(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """Returns the 2D density evaluated at (x, y).

        :param x: x-coordinate(s) where the gradient is evaluated
        :param y: y-coordinate(s) where the gradient is evaluated
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: total 2D surface density
        """
        # Initialize total 2D density
        d_3d = np.zeros_like(x, dtype=float)
        # Loop through each Gaussian component
        for i in range(len(amp)):
            # Add 2D density of ith Gaussian to total
            d_3d += self.gaussian.density_2d(
                x, y, scale_factor * amp[i], sigma[i], center_x, center_y
            )
        return d_3d

    def mass_3d_lens(self, R, amp, sigma, scale_factor=1):
        """Returns the enclosed 3D mass within radius `r`.

        :param R: radial coordinate to evaluate the density
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param scale_factor: global factor applied to each amplitude
        :return: total 3D mass evaluated within `r`
        """
        # Initialize total enclosed mass
        mass_3d = np.zeros_like(R, dtype=float)
        # Loop through each Gaussian component
        for i in range(len(amp)):
            # Add enclosed mass of ith Gaussian to total
            mass_3d += self.gaussian.mass_3d_lens(R, scale_factor * amp[i], sigma[i])
        return mass_3d


class MultiGaussianEllipsePotential(LensProfileBase):
    """Implementation of a sum of elliptical Gaussian lensing potentials.

    Each component is a 2D elliptical Gaussian described by an amplitude and
    width, with ellipticity defined in the potential via parameters `e1` and
    `e2`, which are constant across all components. The Gaussians are centered
    at a common position (`center_x`, `center_y`) and scaled globally using
    `scale_factor`.
    """

    param_names = ["amp", "sigma", "e1", "e2", "center_x", "center_y", "scale_factor"]
    lower_limit_default = {
        "amp": 0,
        "sigma": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
        "scale_factor": 0,
    }
    upper_limit_default = {
        "amp": 100,
        "sigma": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
        "scale_factor": 10000,
    }

    def __init__(self):
        self.gaussian_ellipse_potential = GaussianEllipsePotential()
        super(MultiGaussianEllipsePotential, self).__init__()

    def function(
        self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1
    ):
        """Compute the total lensing potential by summing elliptical Gaussian
        components.

        :param x: x-coordinate(s) where the gradient is evaluated
        :param y: y-coordinate(s) where the gradient is evaluated
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: potential
        """
        f_ = np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            # Add potential of the ith component
            f_ += self.gaussian_ellipse_potential.function(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
            )
        return f_

    def derivatives(
        self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1
    ):
        """Compute the gradient in both angular directions of the total lensing
        potential.

        :param x: x-coordinate(s) where the gradient is evaluated
        :param y: y-coordinate(s) where the gradient is evaluated
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: gradient of potential
        """
        f_x, f_y = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            # Compute gradient of the ith component
            f_x_i, f_y_i = self.gaussian_ellipse_potential.derivatives(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
            )
            f_x += f_x_i
            f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1):
        """Compute the hessian of the total lensing potential.

        :param x: x-coordinate(s) where the gradient is evaluated
        :param y: y-coordinate(s) where the gradient is evaluated
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: hessian of potential
        """
        f_xx, f_yy, f_xy = (
            np.zeros_like(x, dtype=float),
            np.zeros_like(x, dtype=float),
            np.zeros_like(x, dtype=float),
        )
        for i in range(len(amp)):
            # Compute hessian of the ith component
            f_xx_i, f_xy_i, _, f_yy_i = self.gaussian_ellipse_potential.hessian(
                x,
                y,
                amp=scale_factor * amp[i],
                sigma=sigma[i],
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
            )
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_xy, f_xy, f_yy

    def density(self, r, amp, sigma, e1, e2, scale_factor=1):
        """Compute the 3D density at radial distance `r` by summing elliptical
        Gaussians.

        :param r: radial coordinate to evaluate the density
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param scale_factor: global factor applied to each amplitude
        :return: total 3D density evaluated at `r`
        """
        d_ = np.zeros_like(r, dtype=float)
        for i in range(len(amp)):
            # Add density of the ith component
            d_ += self.gaussian_ellipse_potential.density(
                r, scale_factor * amp[i], sigma[i], e1, e2
            )
        return d_

    def density_2d(
        self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1
    ):
        """Returns the 2D density evaluated at (x, y).

        :param x: x-coordinate(s) where the gradient is evaluated
        :param y: y-coordinate(s) where the gradient is evaluated
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param center_x: x-coordinate of the shared center (default is 0)
        :param center_y: y-coordinate of the shared center (default is 0)
        :param scale_factor: global factor applied to each amplitude
        :return: total 2D surface density
        """
        d_3d = np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            # Add 2D density of ith Gaussian to total
            d_3d += self.gaussian_ellipse_potential.density_2d(
                x, y, scale_factor * amp[i], sigma[i], e1, e2, center_x, center_y
            )
        return d_3d

    def mass_3d_lens(self, R, amp, sigma, e1, e2, scale_factor=1):
        """Returns the enclosed 3D mass within radius `r`.

        :param R: radial coordinate to evaluate the density
        :param amp: amplitudes for each Gaussian component
        :param sigma: standard deviations for each component
        :param scale_factor: global factor applied to each amplitude
        :return: total 3D mass evaluated within `r`
        """
        mass_3d = np.zeros_like(R, dtype=float)
        for i in range(len(amp)):
            mass_3d += self.gaussian_ellipse_potential.mass_3d_lens(
                R, scale_factor * amp[i], sigma[i], e1, e2
            )
        return mass_3d
