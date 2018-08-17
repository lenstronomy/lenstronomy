from lenstronomy.LensModel.lens_model import LensModel
import numpy as np
from lenstronomy.Util.util import approx_theta_E

class MultiPlaneLensing(object):

    _no_potential = True

    def __init__(self, full_lensmodel, x_pos, y_pos, lensmodel_params, z_source,
                 z_macro, astropy_instance, macro_indicies, single_background=False):

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

        macromodel_lensmodel, macro_args, halo_lensmodel, halo_args, self._z_background = \
            self._split_lensmodel(full_lensmodel,lensmodel_params,z_break=z_macro,macro_indicies=macro_indicies)
        self._macro_indicies = macro_indicies

        self._foreground = Foreground(halo_lensmodel, self._z_macro, x_pos, y_pos)
        self._halo_args = halo_args

        self._model_to_vary = ToVary(macromodel_lensmodel, self._z_macro)
        self._macro_args = macro_args

        self._single_background = single_background

        self._background = Background(halo_lensmodel, self._z_macro, self._z_source,
                                      single_background=single_background,
                                      approx_Rein = approx_theta_E(self._x_pos, self._y_pos))

    def ray_shooting(self, x, y, kwargs_lens):

        macromodel_args = []

        for ind in self._macro_indicies:
            macromodel_args.append(kwargs_lens[ind])

        # get the deflection angles from foreground and main lens plane subhalos (once)
        x, y, alphax, alphay = self._foreground.ray_shooting(self._halo_args, true_foreground=False,
                                                             offset_index=None, thetax=x, thetay=y
                                                             ,force_compute=True)

        x, y, alphax, alphay = self._model_to_vary.ray_shooting(alphax, alphay, macromodel_args, x, y)

        x_source, y_source = self._background.ray_shooting(alphax, alphay, self._halo_args, x, y,
                                                           offset_index=None,force_compute=True)

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

    def ray_shooting_fast(self, macromodel_args, true_path=True, offset_index=None, thetax=None, thetay=None,
                          force_compute=False):

        # get the deflection angles from foreground and main lens plane subhalos (once)
        x, y, alphax, alphay = self._foreground.ray_shooting(self._halo_args, true_foreground=true_path,
                                                             offset_index=offset_index, thetax=thetax, thetay=thetay
                                                             , force_compute=force_compute)

        x, y, alphax, alphay = self._model_to_vary.ray_shooting(alphax, alphay, macromodel_args, x, y)

        # compute the angular position on the source plane
        x_source, y_source = self._background.ray_shooting(alphax, alphay, self._halo_args, x, y, true_background=true_path,
                                                           offset_index=offset_index)

        betax, betay = x_source * self._T_z_source ** -1, y_source * self._T_z_source ** -1

        if offset_index is None:
            self._beta_x_last, self._beta_y_last = betax, betay

        return betax, betay

    def magnification_fast(self,macromodel_args):

        fxx,fxy,fyx,fyy = self.hessian_fast(macromodel_args)

        det_J = (1-fxx)*(1-fyy)-fyx*fxy

        return np.absolute(det_J**-1)

    def hessian_fast(self,macromodel_args,diff=0.00000001):

        alpha_ra, alpha_dec = self._alpha_fast(self._x_pos, self._y_pos, macromodel_args, true_path=True)

        alpha_ra_dx, alpha_dec_dx = self._alpha_fast(self._x_pos + diff, self._y_pos, macromodel_args, true_path=False,
                                                     offset_index=0)
        alpha_ra_dy, alpha_dec_dy = self._alpha_fast(self._x_pos, self._y_pos + diff, macromodel_args, true_path=False,
                                                     offset_index=1)

        dalpha_rara = (alpha_ra_dx - alpha_ra) * diff ** -1
        dalpha_radec = (alpha_ra_dy - alpha_ra) * diff ** -1
        dalpha_decra = (alpha_dec_dx - alpha_dec) * diff ** -1
        dalpha_decdec = (alpha_dec_dy - alpha_dec) * diff ** -1

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def _alpha_fast(self, x_pos, y_pos, macromodel_args, true_path=False, offset_index = None):

        if offset_index is None:
            return np.array(x_pos - self._beta_x_last), np.array(y_pos - self._beta_y_last)

        beta_x,beta_y = self.ray_shooting_fast(macromodel_args, true_path=true_path, offset_index=offset_index,
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

        return macromodel, macro_args, halo_lensmodel, halo_args, background_z_current

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
        self._diff_rays = [None] * 2

    def ray_shooting(self,args,true_foreground=False,offset_index=None,thetax=None,thetay=None,force_compute=True):

        if true_foreground:

            if not hasattr(self,'_rays'):

                x0, y0 = np.zeros_like(self._x_pos), np.zeros_like(self._y_pos)
                x,y,alphax,alphay = self._halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, self._x_pos, self._y_pos,
                                                                                          z_start=0,
                                                                                          z_stop=self._z_to_vary,
                                                                                          kwargs_lens=args)
                self._rays = {'x':x, 'y':y, 'alphax':alphax, 'alphay':alphay}

            return self._rays['x'], self._rays['y'], self._rays['alphax'], self._rays['alphay']

        elif force_compute:

            x0, y0 = np.zeros_like(thetax), np.zeros_like(thetay)
            x,y,alphax,alphay = self._halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, thetax, thetay,
                                                          z_start=0, z_stop=self._z_to_vary, kwargs_lens=args)
            return x,y,alphax,alphay

        else:

            if self._diff_rays[offset_index] is None:

                x0, y0 = np.zeros_like(self._x_pos), np.zeros_like(self._y_pos)
                x, y, alphax, alphay = self._halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, thetax,
                                                                                             thetay, z_start=0, z_stop=self._z_to_vary, kwargs_lens=args)

                self._diff_rays[offset_index] = {'x': x, 'y': y, 'alphax': alphax, 'alphay': alphay}

            return self._diff_rays[offset_index]['x'], self._diff_rays[offset_index]['y'], self._diff_rays[offset_index]['alphax'], \
                   self._diff_rays[offset_index]['alphay']

class Background(object):

    def __init__(self, background_lensmodel, z_background, z_source, single_background=False, approx_Rein=None):

        self._halos_lensmodel = background_lensmodel
        self._z_background = z_background
        self._z_source = z_source
        self._single_background = single_background
        self._offset_beta = [None,None]

        self._T_main_src = self._halos_lensmodel.lens_model._cosmo_bkg.T_xy(z_background,z_source)
        self._T_z_source = self._halos_lensmodel.lens_model._T_z_source
        self._T_main = self._halos_lensmodel.lens_model._cosmo_bkg.T_xy(0,z_background)
        self._reduced_to_phys_main = self._halos_lensmodel.lens_model._cosmo_bkg.D_xy(0, z_source)*\
                                     self._halos_lensmodel.lens_model._cosmo_bkg.D_xy(z_background, z_source)**-1

        self._approx_deflector = LensModel(lens_model_list=['SIS'], multi_plane=False)

        self._approx_kwargs = [{'theta_E':approx_Rein,'center_x':0, 'center_y':0}]

    def _approx_alpha(self,x,y):

        alphax, alphay = self._approx_deflector.alpha(x*self._T_main**-1,
                                                      y*self._T_main**-1,self._approx_kwargs)

        return -alphax*self._reduced_to_phys_main, -alphay*self._reduced_to_phys_main

    def _fixed_background(self,x_in,y_in,args,alpha_x_approx,alpha_y_approx):

        x, y, _, _ = self._halos_lensmodel.lens_model.ray_shooting_partial(x_in, y_in,alpha_x_approx,alpha_y_approx,
                                                                           self._z_background, self._z_source,args)

        return x, y

    def ray_shooting(self, alphax, alphay, args, x_in, y_in, true_background=True,
                     offset_index=None, force_compute=False):

        if self._single_background:

            x, y = self._ray_shooting_single_background(alphax, alphay, args, x_in, y_in, true_background=true_background,
                                                        offset_index=offset_index,force_compute=force_compute)

        else:

            x, y, _, _ = self._halos_lensmodel.lens_model.ray_shooting_partial(x_in,
                               y_in, alphax, alphay, z_start=self._z_background, z_stop=self._z_source, kwargs_lens=args)

        return x,y

    def _ray_shooting_single_background(self, alphax, alphay, args, x_in, y_in, true_background=True,
                     offset_index=None, force_compute=False):

        if force_compute:

            alpha_x, alpha_y = self._approx_alpha(x_in, y_in)

            _x, _y = self._fixed_background(x_in, y_in, args, alpha_x, alpha_y)

            x = x_in + alphax * self._T_main_src - _x
            y = y_in + alphay * self._T_main_src - _y

        else:

            if not hasattr(self,'_alpha_x_approx'):
                self._alpha_x_approx, self._alpha_y_approx = self._approx_alpha(x_in,y_in)

            if not hasattr(self, '_fixed_beta'):
                self._fixed_beta = {}
                _x, _y = self._fixed_background(x_in, y_in, args, self._alpha_x_approx, self._alpha_y_approx)
                self._fixed_beta['x'] = _x
                self._fixed_beta['y'] = _y

            if true_background:

                x = x_in + alphax * self._T_main_src - self._fixed_beta['x']
                y = y_in + alphay * self._T_main_src - self._fixed_beta['y']

            else:

                if self._offset_beta[offset_index] is None:

                    alpha_x_approx, alpha_y_approx = self._approx_alpha(x_in, y_in)
                    _x, _y = self._fixed_background(x_in, y_in, args, alpha_x_approx, alpha_y_approx)

                    self._offset_beta[offset_index] = {'x': _x, 'y': _y}

                x = x_in + alphax * self._T_main_src - self._offset_beta[offset_index]['x']
                y = y_in + alphay * self._T_main_src - self._offset_beta[offset_index]['y']

        return x,y


