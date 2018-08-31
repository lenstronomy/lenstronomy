from lenstronomy.LensModel.lens_model import LensModel
import numpy as np
from lenstronomy.Util.util import approx_theta_E

class SingleBackground(object):

    _no_potential = True

    def __init__(self, full_lensmodel, x_pos, y_pos, lensmodel_params, z_source,
                 z_macro, astropy_instance, macro_indicies, guess_lensmodel = None,
                 guess_kwargs = None):
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

        self.lensmodel_tovary, _, halo_lensmodel, halo_args, self._z_background = \
            self._split_lensmodel(full_lensmodel, lensmodel_params, z_break=z_macro, macro_indicies=macro_indicies)

        self._halo_args = halo_args
        self._halo_lensmodel = halo_lensmodel

        self._macro_indicies = macro_indicies

        self._T_z_source = full_lensmodel.lens_model._T_z_source
        self._T_main = full_lensmodel.lens_model._cosmo_bkg.T_xy(0, self._z_macro)
        self._T_main_source = full_lensmodel.lens_model._cosmo_bkg.T_xy(self._z_macro, self._z_source)

        self._ray_shoot_init(guess_lensmodel, guess_kwargs)

    def ray_shooting(self, x, y, kwargs_lens):

        macromodel_args = []
        for ind in self._macro_indicies:
            macromodel_args.append(kwargs_lens[ind])

        x0, y0 = np.zeros_like(x), np.zeros_like(y)
        x, y, alphax_foreground, alphay_foreground = self._halo_lensmodel.lens_model.ray_shooting_partial(x0, y0, x, y,
                                                 z_start=0, z_stop=self._z_macro,kwargs_lens = self._halo_args)

        alphax, alphay = self._compute_alpha_macro(x,  y, alphax_foreground, alphay_foreground, macromodel_args)

        delta_beta_x, delta_beta_y = self._compute_deltabeta(x, y)

        betax, betay = self._map_to_source(x, y, alphax, alphay, delta_beta_x, delta_beta_y)

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

        fxx, fxy, fyx, fyy = self.hessian(x,y,kwargs_lens)

        det_J = (1 - fxx) * (1 - fyy) - fxy * fyx

        return det_J ** -1

    def _ray_shooting_fast(self, macromodel_args, offset_index = 0):

        alphax, alphay = self._compute_alpha_macro(self._foreground[offset_index]['x'],
                                          self._foreground[offset_index]['y'],self._foreground[offset_index]['alphax'],
                                                               self._foreground[offset_index]['alphay'],macromodel_args)

        betax, betay = self._map_to_source(self._foreground[offset_index]['x'], self._foreground[offset_index]['y'],
                        alphax, alphay, self._delta_beta[offset_index]['x'], self._delta_beta[offset_index]['y'])

        return betax, betay

    def _hessian_fast(self,macromodel_args,diff=0.00000001):

        alpha_ra, alpha_dec = self._alpha_fast(macromodel_args, 0)
        alpha_ra_dx, alpha_dec_dx = self._alpha_fast(macromodel_args, 1)
        alpha_ra_dy, alpha_dec_dy = self._alpha_fast(macromodel_args, 2)

        dalpha_rara = (alpha_ra_dx - alpha_ra) * diff ** -1
        dalpha_radec = (alpha_ra_dy - alpha_ra) * diff ** -1
        dalpha_decra = (alpha_dec_dx - alpha_dec) * diff ** -1
        dalpha_decdec = (alpha_dec_dy - alpha_dec) * diff ** -1

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def _magnification_fast(self, macromodel_args):

        fxx, fxy, fyx, fyy = self._hessian_fast(macromodel_args)

        det_J = (1 - fxx) * (1 - fyy) - fyx * fxy

        return np.absolute(det_J ** -1)

    def _compute_alpha_macro(self,x,y,alphax_foreground,alphay_foreground,macromodel_args):

        _, _, alphax, alphay = self.lensmodel_tovary.lens_model.ray_shooting_partial(x,y,alphax_foreground,alphay_foreground,
                                                                                     z_start=self._z_macro,z_stop=self._z_macro,
                                                                                     kwargs_lens=macromodel_args,include_z_start=True)
        return alphax, alphay

    def _map_to_source(self, x_in, y_in, alphax, alphay, delta_betax, delta_betay):

        betax = (x_in + alphax * self._T_main_source) * self._T_z_source ** -1 + delta_betax
        betay = (y_in + alphay * self._T_main_source) * self._T_z_source ** -1 + delta_betay

        return betax, betay

    def _alpha_fast(self,macromodel_args,offset_index):

        beta_x, beta_y = self._ray_shooting_fast(macromodel_args, offset_index=offset_index)

        alpha_x = np.array(self.precomputed_theta[offset_index]['x'] - beta_x)
        alpha_y = np.array(self.precomputed_theta[offset_index]['y'] - beta_y)

        return alpha_x, alpha_y

    def _alpha(self, x, y, kwargs_lens):

        beta_x, beta_y = self.ray_shooting(x, y, kwargs_lens)

        alpha_x = np.array(x - beta_x)
        alpha_y = np.array(y - beta_y)

        return alpha_x, alpha_y

    def _alpha_guess(self, thetax, thetay):

        alphax, alphay = self._guess_lensmodel.alpha(thetax, thetay, self._guess_kwargs)

        return alphax * self._guess_red2phys, alphay * self._guess_red2phys

    def _ray_shoot_init(self, guess_lensmodel, guess_kwargs, diff=0.00000001):

        self._init_guess_lensmodel(guess_lensmodel,guess_kwargs)

        # have to do the full ray shooting for three rays
        theta = []
        theta.append({'x':self._x_pos,'y':self._y_pos})
        theta.append({'x':self._x_pos+diff,'y':self._y_pos})
        theta.append({'x':self._x_pos,'y':self._y_pos+diff})

        self.precomputed_theta = theta

        foreground = []
        delta_beta = []

        x0, y0 = np.zeros_like(self._x_pos), np.zeros_like(self._y_pos)

        # first compute the foreground
        for i in range(0,3):

            x, y, alphax, alphay = self._halo_lensmodel.lens_model.ray_shooting_partial(x0, y0, theta[i]['x'],
                                                          theta[i]['y'],z_start=0, z_stop=self._z_macro,
                                                                     kwargs_lens = self._halo_args)

            foreground.append({'x':x,'y':y,'alphax': alphax, 'alphay': alphay, 'thetax':x*self._T_main**-1,
                                    'thetay':y*self._T_main**-1})

            d_betax, d_betay = self._compute_deltabeta(x, y)

            delta_beta.append({'x':d_betax,'y':d_betay})

        self._theta_refx, self._theta_refy = foreground[i]['thetax'], foreground[i]['thetay']
        self._foreground, self._delta_beta = foreground, delta_beta

    def _compute_deltabeta(self, x, y):

        alphax_guess, alphay_guess = self._alpha_guess(x * self._T_main ** -1, y * self._T_main ** -1)

        x_source, y_source, _, _ = self._halo_lensmodel.lens_model.ray_shooting_partial(x, y, alphax_guess,
                                                                                          alphay_guess,
                                                                                          z_start=self._z_macro,
                                                                                          z_stop=self._z_source,
                                                                                          kwargs_lens=self._halo_args)

        betax_straight = (x + alphax_guess * self._T_main_source) * self._T_z_source ** -1
        betay_straight = (y + alphay_guess * self._T_main_source) * self._T_z_source ** -1

        betax = x_source * self._T_z_source ** -1
        betay = y_source * self._T_z_source ** -1

        return betax - betax_straight, betay - betay_straight

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

        return macromodel, macro_args, halo_lensmodel, halo_args, \
               background_z_current

    def _init_guess_lensmodel(self, guess_lensmodel=None, guess_kwargs = None):

        if guess_lensmodel is None:
            self._guess_lensmodel = LensModel(lens_model_list=['SIS'], redshift_list=[self._z_macro],
                                              z_source=self._z_source,
                                              multi_plane=True)

            self._guess_kwargs = [{'theta_E': approx_theta_E(self._x_pos, self._y_pos), 'center_x': 0, 'center_y': 0}]

        else:
            self._guess_lensmodel = guess_lensmodel
            self._guess_kwargs = guess_kwargs

        self._guess_red2phys = self._guess_lensmodel.lens_model._reduced2physical_factor[0]