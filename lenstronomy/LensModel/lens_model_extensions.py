import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.util as util


class LensModelExtensions(LensModel):
    """
    class with extension routines not part of the LensModel core routines
    """

    def magnification_finite(self, x_pos, y_pos, kwargs_lens, kwargs_else=None, source_sigma=0.003, window_size=0.1, grid_number=100,
                             shape="GAUSSIAN"):
        """
        returns the magnification of an extended source with Gaussian light profile
        :param x_pos: x-axis positons of point sources
        :param y_pos: y-axis position of point sources
        :param kwargs_lens: lens model kwargs
        :param kwargs_else: kwargs of image positions
        :param source_sigma: Gaussian sigma in arc sec in source
        :param window_size: size of window to compute the finite flux
        :param grid_number: number of grid cells per axis in the window to numerically comute the flux
        :return: numerically computed brightness of the sources
        """

        mag_finite = np.zeros_like(x_pos)
        deltaPix = float(window_size)/grid_number
        if shape == 'GAUSSIAN':
            from lenstronomy.LightModel.Profiles.gaussian import Gaussian
            quasar = Gaussian()
        elif shape == 'TORUS':
            import lenstronomy.LightModel.Profiles.torus as quasar
        else:
            raise ValueError("shape %s not valid for finite magnification computation!" % shape)
        x_grid, y_grid = util.make_grid(numPix=grid_number, deltapix=deltaPix, subgrid_res=1)
        for i in range(len(x_pos)):
            ra, dec = x_pos[i], y_pos[i]
            center_x, center_y = self.ray_shooting(ra, dec, kwargs_lens, kwargs_else)
            x_source, y_source = self.ray_shooting(x_grid + ra, y_grid + dec, kwargs_lens, kwargs_else)
            I_image = quasar.function(x_source, y_source, 1., source_sigma, source_sigma, center_x, center_y)
            mag_finite[i] = np.sum(I_image) * deltaPix**2
        return mag_finite

    def critical_curve_caustics(self, kwargs_lens, kwargs_else, compute_window=5, grid_scale=0.01):
        """

        :param kwargs_lens: lens model kwargs
        :param kwargs_else: other kwargs
        :param compute_window: window size in arcsec where the critical curve is computed
        :param grid_scale: numerical grid spacing of the computation of the critical curves
        :return: lists of ra and dec arrays corresponding to different disconnected critical curves
        and their caustic counterparts
        """

        numPix = int(compute_window / grid_scale)
        x_grid_high_res, y_grid_high_res = util.make_grid(numPix, deltapix=grid_scale, subgrid_res=1)
        mag_high_res = util.array2image(
            self.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens, kwargs_else))

        #import numpy.ma as ma
        #z = ma.asarray(z, dtype=np.float64)  # Import if want filled contours.

        # Non-filled contours (lines only).
        level = 0.5
        import matplotlib._cntr as cntr
        c = cntr.Cntr(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res)
        nlist = c.trace(level, level, 0)
        segs = nlist[:len(nlist) // 2]
        # print segs  # x,y coords of contour points.

        #cs = ax.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0],
        #                alpha=0.0)
        #paths = cs.collections[0].get_paths()
        paths = segs
        ra_crit_list = []
        dec_crit_list = []
        ra_caustic_list = []
        dec_caustic_list = []
        for p in paths:
            #v = p.vertices
            v = p
            ra_points = v[:, 0]
            dec_points = v[:, 1]
            ra_crit_list.append(ra_points)
            dec_crit_list.append(dec_points)

            ra_caustics, dec_caustics = self.ray_shooting(ra_points, dec_points, kwargs_lens,
                                                                         kwargs_else)
            ra_caustic_list.append(ra_caustics)
            dec_caustic_list.append(dec_caustics)
        return ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list