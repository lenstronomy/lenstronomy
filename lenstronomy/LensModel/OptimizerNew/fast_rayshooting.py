from lenstronomy.LensModel.lens_model import LensModel

class FastRayShooting(object):

    def __init__(self, x_image, y_image, z_lens, z_source, lensModel, param_class, foreground_rays=None):

        self._lensmodel = LensModel
        self._multi_plane = lensModel.lens_model
        self._z_lens = z_lens
        self._z_source = z_source

        self._x_image = x_image
        self._y_image = y_image
        self._param_class = param_class

        self._foreground_rays = foreground_rays

    def ray_shooting(self, x, y, kwargs_lens):

        return self._lensmodel.ray_shooting(x, y, kwargs_lens)

    def ray_shooting_fast(self, args_lens):

        # these do not depend on kwargs_lens_array
        x, y, alpha_x, alpha_y = self._ray_shooting_fast_foreground

        # convert array into new kwargs dictionary
        kwargs_lens = self._param_class.args_to_kwargs(args_lens)
        # evaluate main deflector deflection angles
        x, y, alpha_x, alpha_y = self._multi_plane.ray_shooting_partial(x, y, alpha_x, alpha_y, self._z_lens,
                                                                          self._z_lens, kwargs_lens, include_z_start=True)

        # ray trace through background halos
        x, y, _, _ = self._multi_plane.ray_shooting_partial(
            x, y, alpha_x, alpha_y, self._z_lens, self._z_source, kwargs_lens)

        beta_x, beta_y = self._multi_plane.co_moving2angle_source(x, y)

        return beta_x, beta_y

    def _ray_shooting_fast_foreground(self):

        if self._foreground_rays is None:

            # These do not depend on the kwargs being optimized for
            kwargs_lens = self._param_class.kwargs_lens_init
            x, y, alpha_x, alpha_y = self._multi_plane.ray_shooting_partial(self._x_image, self._y_image, z_start=0.,
                                                         z_stop=self._z_lens, kwargs_lens=kwargs_lens)
            self._foreground_rays = (x, y, alpha_x, alpha_y)

        return self._foreground_rays[0], self._foreground_rays[1], self._foreground_rays[2], self._foreground_rays[3]
