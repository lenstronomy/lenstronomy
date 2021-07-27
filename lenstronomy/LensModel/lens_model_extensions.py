import numpy as np
import lenstronomy.Util.util as util
from skimage.measure import find_contours
from lenstronomy.Util.magnification_finite_util import setup_mag_finite

__all__ = ['LensModelExtensions']


class LensModelExtensions(object):
    """
    class with extension routines not part of the LensModel core routines
    """
    def __init__(self, lensModel):

        """
        :param lensModel: instance of the LensModel() class, or with same functionalities.
        In particular, the following definitions are required to execute all functionalities presented in this class:
        def ray_shooting()
        def magnification()
        def kappa()
        def alpha()
        def hessian()

        """
        self._lensModel = lensModel

    def magnification_finite_adaptive(self, x_image, y_image, source_x, source_y, kwargs_lens,
                                      source_fwhm_parsec, z_source,
                                      cosmo=None, grid_resolution=None,
                                      grid_radius_arcsec=None, axis_ratio=0.5,
                                      tol=0.001, step_size=0.05,
                                      use_largest_eigenvalue=True,
                                      source_light_model='SINGLE_GAUSSIAN',
                                      dx=None, dy=None, size_scale=None, amp_scale=None,
                                      fixed_aperture_size=False):
        """
        This method computes image magnifications with a finite-size background source assuming a Gaussian or a
        double Gaussian source light profile. It can be much faster that magnification_finite for lens models with many
        deflectors and a compact source. This is because most pixels in a rectangular window around a lensed
        image of a compact source do not map onto the source, and therefore don't contribute to the integrated flux in
        the image plane.

        Rather than ray tracing through a rectangular grid, this routine accelerates the computation of image
        magnifications with finite-size sources by ray tracing through an elliptical region oriented such that
        tracks the surface brightness of the lensed image. The aperture size is initially quite small,
        and increases in size until the flux inside of it (and hence the magnification) converges. The orientation of
        the elliptical aperture is computed from the magnification tensor evaluated at the image coordinate.

        If for whatever reason you prefer a circular aperture to the elliptical approximation using the hessian
        eigenvectors, you can just set axis_ratio = 1.

        To use the eigenvalues of the hessian matrix to estimate the optimum axis ratio, set axis_ratio = 0.

        The default settings for the grid resolution and ray tracing window size work well for sources with fwhm between
        0.5 - 100 pc.

        :param x_image: a list or array of x coordinates [units arcsec]
        :param y_image: a list or array of y coordinates [units arcsec]
        :param kwargs_lens: keyword arguments for the lens model
        :param source_fwhm_parsec: the size of the background source [units parsec]
        :param z_source: the source redshift
        :param cosmo: (optional) an instance of astropy.cosmology; if not specified, a default cosmology will be used
        :param grid_resolution: the grid resolution in units arcsec/pixel; if not specified, an appropriate value will
         be estimated from the source size
        :param grid_radius_arcsec: (optional) the size of the ray tracing region in arcsec; if not specified, an appropriate value
         will be estimated from the source size
        :param axis_ratio: the axis ratio of the ellipse used for ray tracing; if axis_ratio = 0, then the eigenvalues
         the hessian matrix will be used to estimate an appropriate axis ratio. Be warned: if the image is highly
         magnified it will tend to curve out of the resulting ellipse
        :param tol: tolerance for convergence in the magnification
        :param step_size: sets the increment for the successively larger ray tracing windows
        :param use_largest_eigenvalue: bool; if True, then the major axis of the ray tracing ellipse region
         will be aligned with the eigenvector corresponding to the largest eigenvalue of the hessian matrix
        :param source_light_model: the model for backgourn source light; currently implemented are 'SINGLE_GAUSSIAN' and
         'DOUBLE_GAUSSIAN'.
        :param dx: used with source model 'DOUBLE_GAUSSIAN', the offset of the second source light profile from the first
         [arcsec]
        :param dy: used with source model 'DOUBLE_GAUSSIAN', the offset of the second source light profile from the first
         [arcsec]
        :param size_scale: used with source model 'DOUBLE_GAUSSIAN', the size of the second source light profile relative
         to the first
        :param amp_scale: used with source model 'DOUBLE_GAUSSIAN', the peak brightness of the second source light profile
         relative to the first
        :param fixed_aperture_size: bool, if True the flux is computed inside a fixed aperture size with radius
         grid_radius_arcsec
        :return: an array of image magnifications
        """

        grid_x_0, grid_y_0, source_model, kwargs_source, grid_resolution, grid_radius_arcsec = setup_mag_finite(cosmo,
                                                                                                                self._lensModel,
                                                                                                                grid_radius_arcsec,
                                                                                                                grid_resolution,
                                                                                                                source_fwhm_parsec,
                                                                                                                source_light_model,
                                                                                                                z_source,
                                                                                                                source_x,
                                                                                                                source_y,
                                                                                                                dx, dy,
                                                                                                                amp_scale,
                                                                                                                size_scale)
        grid_x_0, grid_y_0 = grid_x_0.ravel(), grid_y_0.ravel()

        minimum_magnification = 1e-5

        magnifications = []

        for xi, yi in zip(x_image, y_image):

            w1, w2, v11, v12, v21, v22 = self.hessian_eigenvectors(xi, yi, kwargs_lens)
            _v = [np.array([v11, v12]), np.array([v21, v22])]
            _w = [abs(w1), abs(w2)]
            if use_largest_eigenvalue:
                idx = int(np.argmax(_w))
            else:
                idx = int(np.argmin(_w))
            v = _v[idx]

            rotation_angle = np.arctan(v[1] / v[0]) - np.pi / 2
            grid_x, grid_y = util.rotate(grid_x_0, grid_y_0, rotation_angle)

            if axis_ratio == 0:
                sort = np.argsort(_w)
                q = _w[sort[0]] / _w[sort[1]]
                grid_r = np.hypot(grid_x, grid_y / q).ravel()
            else:
                grid_r = np.hypot(grid_x, grid_y / axis_ratio).ravel()

            flux_array = np.zeros_like(grid_x_0)
            step = step_size * grid_radius_arcsec

            r_min = 0
            if fixed_aperture_size:
                r_max = grid_radius_arcsec
            else:
                r_max = step
            magnification_current = 0.

            while True:

                flux_array = self._magnification_adaptive_iteration(flux_array, xi, yi, grid_x_0, grid_y_0, grid_r,
                                                                    r_min, r_max, self._lensModel, kwargs_lens,
                                                                    source_model, kwargs_source)
                new_magnification = np.sum(flux_array) * grid_resolution ** 2
                diff = abs(new_magnification - magnification_current) / new_magnification

                if r_max >= grid_radius_arcsec:
                    break
                elif diff < tol and new_magnification > minimum_magnification:
                    break
                else:
                    r_min += step
                    r_max += step
                    magnification_current = new_magnification

            magnifications.append(new_magnification)

        return np.array(magnifications)

    @staticmethod
    def _magnification_adaptive_iteration(flux_array, x_image, y_image, grid_x, grid_y, grid_r, r_min, r_max,
                                          lensModel, kwargs_lens, source_model, kwargs_source):
        """
        This function computes the surface brightness of coordinates in 'flux_array' that satisfy r_min < grid_r < r_max,
        where each coordinate in grid_r corresponds to a certain entry in flux_array. Likewise, grid_x, and grid_y

        :param flux_array: an array that contains the flux in each pixel
        :param x_image: image x coordinate
        :param y_image: image y coordinate
        :param grid_x: an array of x coordinates
        :param grid_y: an array of y coordinates
        :param grid_r: an array of projected distances from the origin
        :param r_min: sets the inner radius of the annulus where ray tracing happens
        :param r_max: sets the outer radius of the annulus where ray tracing happens
        :param lensModel: an instance of LensModel
        :param kwargs_lens: keywords for the lens model
        :param source_model: an instance of LightModel
        :param kwargs_source: keywords for the light model
        :return: the flux array where the surface brightness has been computed for all pixels
        with r_min < grid_r < r_max.
        """

        condition1 = grid_r >= r_min
        condition2 = grid_r < r_max
        condition = np.logical_and(condition1, condition2)

        inds = np.where(condition)[0]

        xcoords = grid_x[inds] + x_image
        ycoords = grid_y[inds] + y_image
        beta_x, beta_y = lensModel.ray_shooting(xcoords, ycoords, kwargs_lens)
        flux_in_pixels = source_model.surface_brightness(beta_x, beta_y, kwargs_source)
        flux_array[inds] = flux_in_pixels

        return flux_array

    def magnification_finite(self, x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                             polar_grid=False, aspect_ratio=0.5):
        """
        returns the magnification of an extended source with Gaussian light profile
        :param x_pos: x-axis positons of point sources
        :param y_pos: y-axis position of point sources
        :param kwargs_lens: lens model kwargs
        :param source_sigma: Gaussian sigma in arc sec in source
        :param window_size: size of window to compute the finite flux
        :param grid_number: number of grid cells per axis in the window to numerically compute the flux
        :return: numerically computed brightness of the sources
        """

        mag_finite = np.zeros_like(x_pos)
        deltaPix = float(window_size)/grid_number
        from lenstronomy.LightModel.Profiles.gaussian import Gaussian
        quasar = Gaussian()
        x_grid, y_grid = util.make_grid(numPix=grid_number, deltapix=deltaPix, subgrid_res=1)

        if polar_grid is True:
            a = window_size*0.5
            b = window_size*0.5*aspect_ratio
            ellipse_inds = (x_grid*a**-1) **2 + (y_grid*b**-1) **2 <= 1
            x_grid, y_grid = x_grid[ellipse_inds], y_grid[ellipse_inds]

        for i in range(len(x_pos)):
            ra, dec = x_pos[i], y_pos[i]

            center_x, center_y = self._lensModel.ray_shooting(ra, dec, kwargs_lens)

            if polar_grid is True:
                theta = np.arctan2(dec,ra)
                xcoord, ycoord = util.rotate(x_grid, y_grid, theta)
            else:
                xcoord, ycoord = x_grid, y_grid

            betax, betay = self._lensModel.ray_shooting(xcoord + ra, ycoord + dec, kwargs_lens)

            I_image = quasar.function(betax, betay, 1., source_sigma, center_x, center_y)
            mag_finite[i] = np.sum(I_image) * deltaPix**2
        return mag_finite

    def zoom_source(self, x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                             shape="GAUSSIAN"):
        """
        computes the surface brightness on an image with a zoomed window

        :param x_pos: angular coordinate of center of image
        :param y_pos: angular coordinate of center of image
        :param kwargs_lens: lens model parameter list
        :param source_sigma: source size (in angular units)
        :param window_size: window size in angular units
        :param grid_number: number of grid points per axis
        :param shape: string, shape of source, supports 'GAUSSIAN' and 'TORUS
        :return: 2d numpy array
        """
        deltaPix = float(window_size) / grid_number
        if shape == 'GAUSSIAN':
            from lenstronomy.LightModel.Profiles.gaussian import Gaussian
            quasar = Gaussian()
        elif shape == 'TORUS':
            import lenstronomy.LightModel.Profiles.ellipsoid as quasar
        else:
            raise ValueError("shape %s not valid for finite magnification computation!" % shape)
        x_grid, y_grid = util.make_grid(numPix=grid_number, deltapix=deltaPix, subgrid_res=1)
        center_x, center_y = self._lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        betax, betay = self._lensModel.ray_shooting(x_grid + x_pos, y_grid + y_pos, kwargs_lens)
        image = quasar.function(betax, betay, 1., source_sigma, center_x, center_y)
        return util.array2image(image)

    def critical_curve_tiling(self, kwargs_lens, compute_window=5, start_scale=0.5, max_order=10, center_x=0,
                              center_y=0):
        """

        :param kwargs_lens: lens model keyword argument list
        :param compute_window: total window in the image plane where to search for critical curves
        :param start_scale: float, angular scale on which to start the tiling from (if there are two distinct curves in
         a region, it might only find one.
        :param max_order: int, maximum order in the tiling to compute critical curve triangles
        :param center_x: float, center of the window to compute critical curves and caustics
        :param center_y: float, center of the window to compute critical curves and caustics
        :return: list of positions representing coordinates of the critical curve (in RA and DEC)
        """
        numPix = int(compute_window / start_scale)
        x_grid_init, y_grid_init = util.make_grid(numPix, deltapix=start_scale, subgrid_res=1)
        x_grid_init += center_x
        y_grid_init += center_y
        mag_init = util.array2image(self._lensModel.magnification(x_grid_init, y_grid_init, kwargs_lens))
        x_grid_init = util.array2image(x_grid_init)
        y_grid_init = util.array2image(y_grid_init)

        ra_crit_list = []
        dec_crit_list = []
        # iterate through original triangles and return ra_crit, dec_crit list
        for i in range(numPix-1):
            for j in range(numPix-1):
                edge1 = [x_grid_init[i, j], y_grid_init[i, j], mag_init[i, j]]
                edge2 = [x_grid_init[i+1, j+1], y_grid_init[i+1, j+1], mag_init[i+1, j+1]]
                edge_90_1 = [x_grid_init[i, j+1], y_grid_init[i, j+1], mag_init[i, j+1]]
                edge_90_2 = [x_grid_init[i+1, j], y_grid_init[i+1, j], mag_init[i+1, j]]
                ra_crit, dec_crit = self._tiling_crit(edge1, edge2, edge_90_1, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_list += ra_crit  # list addition
                dec_crit_list += dec_crit  # list addition
                ra_crit, dec_crit = self._tiling_crit(edge1, edge2, edge_90_2, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_list += ra_crit  # list addition
                dec_crit_list += dec_crit  # list addition
        return np.array(ra_crit_list), np.array(dec_crit_list)

    def _tiling_crit(self, edge1, edge2, edge_90, max_order, kwargs_lens):
        """
        tiles a rectangular triangle and compares the signs of the magnification

        :param edge1: [ra_coord, dec_coord, magnification]
        :param edge2: [ra_coord, dec_coord, magnification]
        :param edge_90: [ra_coord, dec_coord, magnification]
        :param max_order: maximal order to fold triangle
        :param kwargs_lens: lens model keyword argument list
        :return:
        """
        ra_1, dec_1, mag_1 = edge1
        ra_2, dec_2, mag_2 = edge2
        ra_3, dec_3, mag_3 = edge_90
        sign_list = np.sign([mag_1, mag_2, mag_3])
        if sign_list[0] == sign_list[1] and sign_list[0] == sign_list[2]:  # if all signs are the same
            return [], []
        else:
            # split triangle along the long axis
            # execute tiling twice
            # add ra_crit and dec_crit together
            # if max depth has been reached, return the mean value in the triangle
            max_order -= 1
            if max_order <= 0:
                return [(ra_1 + ra_2 + ra_3)/3], [(dec_1 + dec_2 + dec_3)/3]
            else:
                # split triangle
                ra_90_ = (ra_1 + ra_2)/2  # find point in the middle of the long axis to split triangle
                dec_90_ = (dec_1 + dec_2)/2
                mag_90_ = self._lensModel.magnification(ra_90_, dec_90_, kwargs_lens)
                edge_90_ = [ra_90_, dec_90_, mag_90_]
                ra_crit, dec_crit = self._tiling_crit(edge1=edge_90, edge2=edge1, edge_90=edge_90_, max_order=max_order,
                                                      kwargs_lens=kwargs_lens)
                ra_crit_2, dec_crit_2 = self._tiling_crit(edge1=edge_90, edge2=edge2, edge_90=edge_90_, max_order=max_order,
                                                          kwargs_lens=kwargs_lens)
                ra_crit += ra_crit_2
                dec_crit += dec_crit_2
                return ra_crit, dec_crit

    def critical_curve_caustics(self, kwargs_lens, compute_window=5, grid_scale=0.01, center_x=0, center_y=0):
        """

        :param kwargs_lens: lens model kwargs
        :param compute_window: window size in arcsec where the critical curve is computed
        :param grid_scale: numerical grid spacing of the computation of the critical curves
        :param center_x: float, center of the window to compute critical curves and caustics
        :param center_y: float, center of the window to compute critical curves and caustics
        :return: lists of ra and dec arrays corresponding to different disconnected critical curves and their caustic counterparts

        """
        numPix = int(compute_window / grid_scale)
        x_grid_high_res, y_grid_high_res = util.make_grid(numPix, deltapix=grid_scale, subgrid_res=1)
        x_grid_high_res += center_x
        y_grid_high_res += center_y
        mag_high_res = util.array2image(self._lensModel.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens))

        ra_crit_list = []
        dec_crit_list = []
        ra_caustic_list = []
        dec_caustic_list = []

        paths = find_contours(1/mag_high_res, 0.)
        for i, v in enumerate(paths):
            # x, y changed because of skimage conventions
            ra_points = v[:, 1] * grid_scale - grid_scale * (numPix-1)/2 + center_x
            dec_points = v[:, 0] * grid_scale - grid_scale * (numPix-1)/2 + center_y
            ra_crit_list.append(ra_points)
            dec_crit_list.append(dec_points)
            ra_caustics, dec_caustics = self._lensModel.ray_shooting(ra_points, dec_points, kwargs_lens)
            ra_caustic_list.append(ra_caustics)
            dec_caustic_list.append(dec_caustics)
        return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list

    def hessian_eigenvectors(self, x, y, kwargs_lens, diff=None):
        """
        computes magnification eigenvectors at position (x, y)

        :param x: x-position
        :param y: y-position
        :param kwargs_lens: lens model keyword arguments
        :return: radial stretch, tangential stretch
        """
        f_xx, f_xy, f_yx, f_yy = self._lensModel.hessian(x, y, kwargs_lens, diff=diff)
        if isinstance(x, int) or isinstance(x, float):
            A = np.array([[1-f_xx, f_xy], [f_yx, 1-f_yy]])
            w, v = np.linalg.eig(A)
            v11, v12, v21, v22 = v[0, 0], v[0, 1], v[1, 0], v[1, 1]
            w1, w2 = w[0], w[1]
        else:
            w1, w2, v11, v12, v21, v22 = np.empty(len(x), dtype=float), np.empty(len(x), dtype=float), np.empty_like(x), np.empty_like(x), np.empty_like(x), np.empty_like(x)
            for i in range(len(x)):
                A = np.array([[1 - f_xx[i], f_xy[i]], [f_yx[i], 1 - f_yy[i]]])
                w, v = np.linalg.eig(A)
                w1[i], w2[i] = w[0], w[1]
                v11[i], v12[i], v21[i], v22[i] = v[0, 0], v[0, 1], v[1, 0], v[1, 1]
        return w1, w2, v11, v12, v21, v22

    def radial_tangential_stretch(self, x, y, kwargs_lens, diff=None, ra_0=0, dec_0=0,
                                  coordinate_frame_definitions=False):
        """
        computes the radial and tangential stretches at a given position

        :param x: x-position
        :param y: y-position
        :param kwargs_lens: lens model keyword arguments
        :param diff: float or None, finite average differential scale
        :return: radial stretch, tangential stretch
        """
        w1, w2, v11, v12, v21, v22 = self.hessian_eigenvectors(x, y, kwargs_lens, diff=diff)
        v_x, v_y = x - ra_0, y - dec_0

        prod_v1 = v_x*v11 + v_y*v12
        prod_v2 = v_x*v21 + v_y*v22
        if isinstance(x, int) or isinstance(x, float):
            if (coordinate_frame_definitions is True and abs(prod_v1) >= abs(prod_v2)) or (coordinate_frame_definitions is False and w1 >= w2):
            #if w1 > w2:
            #if abs(prod_v1) > abs(prod_v2):  # radial vector has larger scalar product to the zero point
                lambda_rad = 1. / w1
                lambda_tan = 1. / w2
                v1_rad, v2_rad = v11, v12
                v1_tan, v2_tan = v21, v22
                prod_r = prod_v1
            else:
                lambda_rad = 1. / w2
                lambda_tan = 1. / w1
                v1_rad, v2_rad = v21, v22
                v1_tan, v2_tan = v11, v12
                prod_r = prod_v2
            if prod_r < 0:  # if radial eigenvector points towards the center
                v1_rad, v2_rad = -v1_rad, -v2_rad
            if v1_rad * v2_tan - v2_rad * v1_tan < 0:  # cross product defines orientation of the tangential eigenvector
                v1_tan *= -1
                v2_tan *= -1

        else:
            lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan = np.empty(len(x), dtype=float), np.empty(len(x), dtype=float), np.empty_like(x), np.empty_like(x), np.empty_like(x), np.empty_like(x)
            for i in range(len(x)):
                if (coordinate_frame_definitions is True and abs(prod_v1[i]) >= abs(prod_v2[i])) or (
                        coordinate_frame_definitions is False and w1[i] >= w2[i]):
                #if w1[i] > w2[i]:
                    lambda_rad[i] = 1. / w1[i]
                    lambda_tan[i] = 1. / w2[i]
                    v1_rad[i], v2_rad[i] = v11[i], v12[i]
                    v1_tan[i], v2_tan[i] = v21[i], v22[i]
                    prod_r = prod_v1[i]
                else:
                    lambda_rad[i] = 1. / w2[i]
                    lambda_tan[i] = 1. / w1[i]
                    v1_rad[i], v2_rad[i] = v21[i], v22[i]
                    v1_tan[i], v2_tan[i] = v11[i], v12[i]
                    prod_r = prod_v2[i]
                if prod_r < 0:  # if radial eigenvector points towards the center
                    v1_rad[i], v2_rad[i] = -v1_rad[i], -v2_rad[i]
                if v1_rad[i] * v2_tan[i] - v2_rad[i] * v1_tan[i] < 0:  # cross product defines orientation of the tangential eigenvector
                    v1_tan[i] *= -1
                    v2_tan[i] *= -1

        return lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan

    def radial_tangential_differentials(self, x, y, kwargs_lens, center_x=0, center_y=0, smoothing_3rd=0.001,
                                        smoothing_2nd=None):
        """
        computes the differentials in stretches and directions

        :param x: x-position
        :param y: y-position
        :param kwargs_lens: lens model keyword arguments
        :param center_x: x-coord of center towards which the rotation direction is defined
        :param center_y: x-coord of center towards which the rotation direction is defined
        :param smoothing_3rd: finite differential length of third order in units of angle
        :param smoothing_2nd: float or None, finite average differential scale of Hessian
        :return:
        """
        lambda_rad, lambda_tan, v1_rad, v2_rad, v1_tan, v2_tan = self.radial_tangential_stretch(x, y, kwargs_lens,
                                                                                                diff=smoothing_2nd,
                                                                                                ra_0=center_x, dec_0=center_y,
                                                                                                coordinate_frame_definitions=True)
        x0 = x - center_x
        y0 = y - center_y

        # computing angle of tangential vector in regard to the defined coordinate center
        cos_angle = (v1_tan * x0 + v2_tan * y0) / np.sqrt((x0 ** 2 + y0 ** 2) * (v1_tan ** 2 + v2_tan ** 2))# * np.sign(v1_tan * y0 - v2_tan * x0)
        orientation_angle = np.arccos(cos_angle) - np.pi / 2

        # computing differentials in tangential and radial directions
        dx_tan = x + smoothing_3rd * v1_tan
        dy_tan = y + smoothing_3rd * v2_tan
        lambda_rad_dtan, lambda_tan_dtan, v1_rad_dtan, v2_rad_dtan, v1_tan_dtan, v2_tan_dtan = self.radial_tangential_stretch(dx_tan, dy_tan, kwargs_lens, diff=smoothing_2nd,
                                                                                                                              ra_0=center_x, dec_0=center_y, coordinate_frame_definitions=True)
        dx_rad = x + smoothing_3rd * v1_rad
        dy_rad = y + smoothing_3rd * v2_rad
        lambda_rad_drad, lambda_tan_drad, v1_rad_drad, v2_rad_drad, v1_tan_drad, v2_tan_drad = self.radial_tangential_stretch(
            dx_rad, dy_rad, kwargs_lens, diff=smoothing_2nd, ra_0=center_x, dec_0=center_y, coordinate_frame_definitions=True)

        # eigenvalue differentials in tangential and radial direction
        dlambda_tan_dtan = (lambda_tan_dtan - lambda_tan) / smoothing_3rd# * np.sign(v1_tan * y0 - v2_tan * x0)
        dlambda_tan_drad = (lambda_tan_drad - lambda_tan) / smoothing_3rd# * np.sign(v1_rad * x0 + v2_rad * y0)
        dlambda_rad_drad = (lambda_rad_drad - lambda_rad) / smoothing_3rd# * np.sign(v1_rad * x0 + v2_rad * y0)
        dlambda_rad_dtan = (lambda_rad_dtan - lambda_rad) / smoothing_3rd# * np.sign(v1_rad * x0 + v2_rad * y0)

        # eigenvector direction differentials in tangential and radial direction
        cos_dphi_tan_dtan = v1_tan * v1_tan_dtan + v2_tan * v2_tan_dtan #/ (np.sqrt(v1_tan**2 + v2_tan**2) * np.sqrt(v1_tan_dtan**2 + v2_tan_dtan**2))
        norm = np.sqrt(v1_tan**2 + v2_tan**2) * np.sqrt(v1_tan_dtan**2 + v2_tan_dtan**2)
        cos_dphi_tan_dtan /= norm
        arc_cos_dphi_tan_dtan = np.arccos(np.abs(np.minimum(cos_dphi_tan_dtan, 1)))
        dphi_tan_dtan = arc_cos_dphi_tan_dtan / smoothing_3rd

        cos_dphi_tan_drad = v1_tan * v1_tan_drad + v2_tan * v2_tan_drad  # / (np.sqrt(v1_tan ** 2 + v2_tan ** 2) * np.sqrt(v1_tan_drad ** 2 + v2_tan_drad ** 2))
        norm = np.sqrt(v1_tan ** 2 + v2_tan ** 2) * np.sqrt(v1_tan_drad ** 2 + v2_tan_drad ** 2)
        cos_dphi_tan_drad /= norm
        arc_cos_dphi_tan_drad = np.arccos(np.abs(np.minimum(cos_dphi_tan_drad, 1)))
        dphi_tan_drad = arc_cos_dphi_tan_drad / smoothing_3rd

        cos_dphi_rad_drad = v1_rad * v1_rad_drad + v2_rad * v2_rad_drad #/ (np.sqrt(v1_rad**2 + v2_rad**2) * np.sqrt(v1_rad_drad**2 + v2_rad_drad**2))
        norm = np.sqrt(v1_rad**2 + v2_rad**2) * np.sqrt(v1_rad_drad**2 + v2_rad_drad**2)
        cos_dphi_rad_drad /= norm
        cos_dphi_rad_drad = np.minimum(cos_dphi_rad_drad, 1)
        dphi_rad_drad = np.arccos(cos_dphi_rad_drad) / smoothing_3rd

        cos_dphi_rad_dtan = v1_rad * v1_rad_dtan + v2_rad * v2_rad_dtan # / (np.sqrt(v1_rad ** 2 + v2_rad ** 2) * np.sqrt(v1_rad_dtan ** 2 + v2_rad_dtan ** 2))
        norm = np.sqrt(v1_rad ** 2 + v2_rad ** 2) * np.sqrt(v1_rad_dtan ** 2 + v2_rad_dtan ** 2)
        cos_dphi_rad_dtan /= norm
        cos_dphi_rad_dtan = np.minimum(cos_dphi_rad_dtan, 1)
        dphi_rad_dtan = np.arccos(cos_dphi_rad_dtan) / smoothing_3rd

        return lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan

    def curved_arc_estimate(self, x, y, kwargs_lens, smoothing=None, smoothing_3rd=0.001, tan_diff=False):
        """
        performs the estimation of the curved arc description at a particular position of an arbitrary lens profile

        :param x: float, x-position where the estimate is provided
        :param y: float, y-position where the estimate is provided
        :param kwargs_lens: lens model keyword arguments
        :param smoothing: (optional) finite differential of second derivative (radial and tangential stretches)
        :param smoothing_3rd: differential scale for third derivative to estimate the tangential curvature
        :param tan_diff: boolean, if True, also returns the relative tangential stretch differential in tangential direction
        :return: keyword argument list corresponding to a CURVED_ARC profile at (x, y) given the initial lens model
        """
        radial_stretch, tangential_stretch, v_rad1, v_rad2, v_tang1, v_tang2 = self.radial_tangential_stretch(x, y, kwargs_lens, diff=smoothing)
        dx_tang = x + smoothing_3rd * v_tang1
        dy_tang = y + smoothing_3rd * v_tang2
        _, _, _, _, v_tang1_dt, v_tang2_dt = self.radial_tangential_stretch(dx_tang, dy_tang,kwargs_lens,
                                                                            diff=smoothing)
        d_tang1 = v_tang1_dt - v_tang1
        d_tang2 = v_tang2_dt - v_tang2
        delta = np.sqrt(d_tang1**2 + d_tang2**2)
        if delta > 1:
            d_tang1 = v_tang1_dt + v_tang1
            d_tang2 = v_tang2_dt + v_tang2
            delta = np.sqrt(d_tang1 ** 2 + d_tang2 ** 2)
        curvature = delta / smoothing_3rd
        direction = np.arctan2(v_rad2 * np.sign(v_rad1 * x + v_rad2 * y), v_rad1 * np.sign(v_rad1 * x + v_rad2 * y))

        kwargs_arc = {'radial_stretch': radial_stretch,
                      'tangential_stretch': tangential_stretch,
                      'curvature': curvature,
                      'direction': direction,
                      'center_x': x, 'center_y': y}
        if tan_diff:
            lambda_rad, lambda_tan, orientation_angle, dlambda_tan_dtan, dlambda_tan_drad, dlambda_rad_drad, dlambda_rad_dtan, dphi_tan_dtan, dphi_tan_drad, dphi_rad_drad, dphi_rad_dtan = self.radial_tangential_differentials(x, y, kwargs_lens, center_x=0, center_y=0, smoothing_3rd=smoothing_3rd)
            kwargs_arc['dtan_dtan'] = dlambda_tan_dtan / lambda_tan
        return kwargs_arc

    def tangential_average(self, x, y, kwargs_lens, dr, smoothing=None, num_average=9):
        """
        computes average tangential stretch around position (x, y) within dr in radial direction

        :param x: x-position (float)
        :param y: y-position (float)
        :param kwargs_lens: lens model keyword argument list
        :param dr: averaging scale in radial direction
        :param smoothing: smoothing scale of derivative
        :param num_average: integer, number of points averaged over within dr in the radial direction
        :return:
        """
        radial_stretch, tangential_stretch, v_rad1, v_rad2, v_tang1, v_tang2 = self.radial_tangential_stretch(x, y,
                                                                                                              kwargs_lens,
                                                                                                              diff=smoothing)
        dr_array = np.linspace(start=-dr/2., stop=dr/2., num=num_average)
        dx_r = x + dr_array * v_rad1
        dy_r = y + dr_array * v_rad2
        _, tangential_stretch_dr, _, _, _, _ = self.radial_tangential_stretch(dx_r, dy_r, kwargs_lens, diff=smoothing)
        return np.average(tangential_stretch_dr)

    def curved_arc_finite_area(self, x, y, kwargs_lens, dr):
        """
        computes an estimated curved arc over a finite extent mimicking the appearance of a finite source with radius dr

        :param x: x-position (float)
        :param y: y-position (float)
        :param kwargs_lens: lens model keyword argument list
        :param dr: radius of finite source
        :return: keyword arguments of curved arc
        """

        # estimate curvature centroid as the median around the circle

        # make circle of points around position of interest
        x_c, y_c = util.points_on_circle(radius=dr, num_points=20, connect_ends=False)

        c_x_list, c_y_list = [], []
        # loop through curved arc estimate and compute curvature centroid
        for x_, y_ in zip(x_c, y_c):
            kwargs_arc_ = self.curved_arc_estimate(x_, y_, kwargs_lens)
            direction = kwargs_arc_['direction']
            curvature = kwargs_arc_['curvature']
            center_x = x_ - np.cos(direction) / curvature
            center_y = y_ - np.sin(direction) / curvature
            c_x_list.append(center_x)
            c_y_list.append(center_y)
        center_x, center_y = np.median(c_x_list), np.median(c_y_list)

        # compute curvature and direction to the average centroid from the position of interest
        r = np.sqrt((x - center_x) ** 2 + (y - center_y)**2)
        curvature = 1 / r
        direction = np.arctan2(y - center_y, x - center_x)

        # compute average radial stretch as the inverse difference in the source position
        x_r = x + np.cos(direction) * dr
        y_r = y + np.sin(direction) * dr
        x_r_ = x - np.cos(direction) * dr
        y_r_ = y - np.sin(direction) * dr

        xs_r, ys_r = self._lensModel.ray_shooting(x_r, y_r, kwargs_lens)
        xs_r_, ys_r_ = self._lensModel.ray_shooting(x_r_, y_r_, kwargs_lens)
        ds = np.sqrt((xs_r - xs_r_)**2 + (ys_r - ys_r_)**2)
        radial_stretch = (2 * dr) / ds

        # compute average tangential stretch as the inverse difference in the sosurce position
        x_t = x - np.sin(direction) * dr
        y_t = y + np.cos(direction) * dr
        x_t_ = x + np.sin(direction) * dr
        y_t_ = y - np.cos(direction) * dr

        xs_t, ys_t = self._lensModel.ray_shooting(x_t, y_t, kwargs_lens)
        xs_t_, ys_t_ = self._lensModel.ray_shooting(x_t_, y_t_, kwargs_lens)
        ds = np.sqrt((xs_t - xs_t_) ** 2 + (ys_t - ys_t_) ** 2)
        tangential_stretch = (2 * dr) / ds
        kwargs_arc = {'direction': direction, 'radial_stretch': radial_stretch,
                      'tangential_stretch': tangential_stretch, 'center_x': x, 'center_y': y,
                      'curvature': curvature}
        return kwargs_arc
