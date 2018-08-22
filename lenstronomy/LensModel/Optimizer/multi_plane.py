from lenstronomy.LensModel.lens_model import LensModel
import numpy as np
from lenstronomy.Util.util import approx_theta_E

class MultiPlaneLensing(object):

    _no_potential = True

    def __init__(self, full_lensmodel, x_pos, y_pos, lensmodel_params, z_source,
                 z_macro, astropy_instance, macro_indicies):

        """
        This class performs (fast) lensing computations for multi-plane lensing scenarios
        :param full_lensmodel:
        :param x_pos:
        :param y_pos:
        :param lensmodel_params:
        :param z_source:
        :param z_macro:
        :param astropy_instance:
        :param macro_indicies:
        """

        self._z_macro, self._z_source = z_macro, z_source

        self._astropy_instance = astropy_instance

        self._x_pos, self._y_pos = np.array(x_pos), np.array(y_pos)
        self._nimg = len(x_pos)
        self._mag_idx = 0

        self._full_lensmodel, self._lensmodel_params = full_lensmodel, lensmodel_params

        self._T_z_source = full_lensmodel.lens_model._T_z_source

        macromodel_lensmodel, macro_args, halo_lensmodel, halo_args, front_lensmodel, front_args, self._z_background = \
            self._split_lensmodel(full_lensmodel,lensmodel_params,z_break=z_macro,macro_indicies=macro_indicies)
        self._macro_indicies = macro_indicies

        self._front_lensmodel = front_lensmodel
        self._front_args = front_args

        self._foreground = Foreground(halo_lensmodel, self._z_macro, x_pos, y_pos)
        self._halo_args = halo_args

        self._model_to_vary = ToVary(macromodel_lensmodel, self._z_macro)
        self._macro_args = macro_args

        self._background = Background(halo_lensmodel, self._z_macro, self._z_source)

    def ray_shooting(self, x, y, kwargs_lens):

        macromodel_args = []

        for ind in self._macro_indicies:
            macromodel_args.append(kwargs_lens[ind])

        # get the deflection angles from foreground and main lens plane subhalos (once)
        x, y, alphax, alphay = self._foreground.ray_shooting(self._halo_args, thetax=x, thetay=y,
                                                             force_compute=True)

        x, y, alphax, alphay = self._model_to_vary.ray_shooting(alphax, alphay, macromodel_args, x, y)

        x_source, y_source = self._background.ray_shooting(alphax, alphay, self._halo_args, x, y)

        betax, betay = x_source * self._T_z_source ** -1, y_source * self._T_z_source ** -1

        return betax, betay

    def hessian(self, x, y, kwargs_lens, diff=0.00000001):

        alpha_ra, alpha_dec = self._alpha(x, y, kwargs_lens)

        alpha_ra_dx, alpha_dec_dx = self._alpha(x + diff, y, kwargs_lens)
        alpha_ra_dy, alpha_dec_dy = self._alpha(x, y + diff, kwargs_lens)

        dalpha_rara = (alpha_ra_dx - alpha_ra) * diff ** -1
        dalpha_radec = (alpha_ra_dy - alpha_ra) * diff ** -1
        dalpha_decra = (alpha_dec_dx - alpha_dec) * diff ** -1
        dalpha_decdec = (alpha_dec_dy - alpha_dec) * diff ** -1

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def magnification(self,x,y,kwargs_lens):

        f_xx, f_xy, f_yx, f_yy = self.hessian(x,y,kwargs_lens)

        det_A = (1 - f_xx) * (1 - f_yy) - f_xy * f_yx

        return det_A**-1

    def _ray_shooting_fast(self, macromodel_args, offset_index=0, thetax=None, thetay=None,
                           force_compute=False):

        # get the deflection angles from foreground and main lens plane subhalos (once)
        x, y, alphax, alphay = self._foreground.ray_shooting(self._halo_args,offset_index=offset_index,
                                                             thetax=thetax, thetay=thetay,
                                                             force_compute=force_compute)

        x, y, alphax, alphay = self._model_to_vary.ray_shooting(alphax, alphay, macromodel_args, x, y)


        x_source, y_source = self._background.ray_shooting(alphax, alphay, self._halo_args, x, y)

        betax, betay = x_source * self._T_z_source ** -1, y_source * self._T_z_source ** -1

        if offset_index == 0:
            self._beta_x_last, self._beta_y_last = betax, betay

        return betax, betay

    def _magnification_fast(self, macromodel_args):

        fxx,fxy,fyx,fyy = self._hessian_fast(macromodel_args)

        det_J = (1-fxx)*(1-fyy)-fyx*fxy

        return np.absolute(det_J**-1)

    def _hessian_fast(self, macromodel_args, diff=0.00000001):

        alpha_ra, alpha_dec = self._alpha_fast(self._x_pos, self._y_pos, macromodel_args)

        alpha_ra_dx, alpha_dec_dx = self._alpha_fast(self._x_pos + diff, self._y_pos, macromodel_args,
                                                     offset_index=1)
        alpha_ra_dy, alpha_dec_dy = self._alpha_fast(self._x_pos, self._y_pos + diff, macromodel_args,
                                                     offset_index=2)

        dalpha_rara = (alpha_ra_dx - alpha_ra) * diff ** -1
        dalpha_radec = (alpha_ra_dy - alpha_ra) * diff ** -1
        dalpha_decra = (alpha_dec_dx - alpha_dec) * diff ** -1
        dalpha_decdec = (alpha_dec_dy - alpha_dec) * diff ** -1

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def _alpha_fast(self, x_pos, y_pos, macromodel_args, offset_index = 0):

        if offset_index == 0 and hasattr(self,'_beta_x_last'):
            return np.array(x_pos - self._beta_x_last), np.array(y_pos - self._beta_y_last)

        beta_x,beta_y = self._ray_shooting_fast(macromodel_args, offset_index=offset_index,
                                                thetax=x_pos, thetay=y_pos)

        alpha_x = np.array(x_pos - beta_x)
        alpha_y = np.array(y_pos - beta_y)

        return alpha_x, alpha_y

    def _alpha(self, x_pos, y_pos, kwargs_lens):

        beta_x,beta_y = self.ray_shooting(x_pos, y_pos, kwargs_lens)

        alpha_x = np.array(x_pos - beta_x)
        alpha_y = np.array(y_pos - beta_y)

        return alpha_x, alpha_y

    def _split_lensmodel(self, lensmodel, lensmodel_args, z_break, macro_indicies):

        """

        :param lensmodel: lensmodel to break up
        :param lensmodel_args: kwargs to break up
        :param z_break: the break redshift
        :param macro_indicies: the indicies of the macromodel in the lens model list
        :return: instances of LensModel for foreground, main lens plane and background halos, and the macromodel
        """

        front_model_names, front_redshifts, front_args = [], [], []
        back_model_names, back_redshifts, back_args = [], [], []
        macro_names, macro_redshifts, macro_args = [], [], []

        halo_names, halo_redshifts, halo_args = [], [], []

        background_z_current = self._z_macro + 0.5 * (self._z_source - self._z_macro)

        for i in range(0, len(lensmodel.lens_model_list)):

            z = lensmodel.redshift_list[i]

            if i not in macro_indicies:

                halo_names.append(lensmodel.lens_model_list[i])
                halo_redshifts.append(z)
                halo_args.append(lensmodel_args[i])

                if z > z_break:

                    if z < background_z_current:
                        background_z_current = z

                    back_model_names.append(lensmodel.lens_model_list[i])
                    back_redshifts.append(z)
                    back_args.append(lensmodel_args[i])

                elif z <= z_break:
                    front_model_names.append(lensmodel.lens_model_list[i])
                    front_redshifts.append(z)
                    front_args.append(lensmodel_args[i])

            else:

                macro_names.append(lensmodel.lens_model_list[i])
                macro_redshifts.append(z)
                macro_args.append(lensmodel_args[i])

        macromodel = LensModel(lens_model_list=macro_names, redshift_list=macro_redshifts, cosmo=self._astropy_instance,
                               multi_plane=True,
                               z_source=self._z_source)

        halo_lensmodel = LensModel(lens_model_list=front_model_names+back_model_names, redshift_list=front_redshifts+back_redshifts,
                                   cosmo=self._astropy_instance, multi_plane=True, z_source=self._z_source)
        halo_args = front_args+back_args

        front_lensmodel = LensModel(lens_model_list=front_model_names+macro_names,
                                    redshift_list=front_redshifts+macro_redshifts,
                                    cosmo=self._astropy_instance,
                                    multi_plane=True,z_source=self._z_source)

        return macromodel, macro_args, halo_lensmodel, halo_args, front_lensmodel,front_args,\
               background_z_current

class ToVary(object):

    def __init__(self,tovary_lensmodel,z_to_vary):

        self._tovary_lensmodel = tovary_lensmodel
        self._z_to_vary = z_to_vary

    def ray_shooting(self, thetax, thetay, args, x_in, y_in):

        x, y, alphax, alphay = self._tovary_lensmodel.lens_model. \
            ray_shooting_partial(x_in, y_in, thetax, thetay, z_start=self._z_to_vary,
                                 z_stop=self._z_to_vary, kwargs_lens=args, include_z_start=True)

        return x, y, alphax, alphay

class Foreground(object):

    def __init__(self, foreground_lensmodel, z_to_vary, x_pos, y_pos):

        self._halos_lensmodel = foreground_lensmodel
        self._z_to_vary = z_to_vary
        self._x_pos, self._y_pos = x_pos, y_pos
        self._rays = [None] * 3

    def ray_shooting(self,args,offset_index=None,thetax=None,thetay=None,force_compute=True):

        if force_compute:
            x0, y0 = np.zeros_like(thetax), np.zeros_like(thetay)
            x, y, alphax, alphay = self._halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, thetax, thetay,
                                                                                         z_start=0,
                                                                                         z_stop=self._z_to_vary,
                                                                                         kwargs_lens=args)
            return x, y, alphax, alphay

        else:

            if self._rays[offset_index] is None:
                x0, y0 = np.zeros_like(self._x_pos), np.zeros_like(self._y_pos)

                if offset_index == 0:
                    thetax, thetay = self._x_pos, self._y_pos

                x, y, alphax, alphay = self._halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, thetax,
                                                                                             thetay, z_start=0,
                                                                                             z_stop=self._z_to_vary,
                                                                                             kwargs_lens=args)

                self._rays[offset_index] = {'x': x, 'y': y, 'alphax': alphax, 'alphay': alphay}

            return self._rays[offset_index]['x'], self._rays[offset_index]['y'],\
                   self._rays[offset_index]['alphax'], self._rays[offset_index]['alphay']

class Background(object):

    def __init__(self, background_lensmodel, z_background, z_source):

        self._halos_lensmodel = background_lensmodel
        self._z_background = z_background
        self._z_source = z_source

    def ray_shooting(self, alphax, alphay, args, x_in, y_in):

        x, y, _, _ = self._halos_lensmodel.lens_model.ray_shooting_partial(x_in, y_in, alphax, alphay,
                                                      self._z_background, self._z_source, args)

        return x,y
