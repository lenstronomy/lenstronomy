from lenstronomy.LensModel.lens_model import LensModel
import numpy as np


class MultiPlaneLensing(object):

    _no_potential = True

    def __init__(self, full_lensmodel, x_pos, y_pos, lensmodel_params, z_source,
                 z_macro, astropy_instance, macro_indicies, optimizer_kwargs, numerical_alpha_class,
                 observed_convention_index=None):

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

        self._T_z_source = full_lensmodel.lens_model._multi_plane_base._T_z_source

        self._observed_convention_index = observed_convention_index

        macromodel_lensmodel, macro_args, halo_lensmodel, halo_args = \
            self._split_lensmodel(full_lensmodel,lensmodel_params,macro_indicies,numerical_alpha_class,
                                  observed_convention_index)
        self._macro_indicies = macro_indicies

        if 'precomputed_rays' in optimizer_kwargs:
            self._foreground = Foreground(halo_lensmodel, macromodel_lensmodel, self._z_macro, x_pos, y_pos,
                                          precompupted_rays=optimizer_kwargs['precomputed_rays'])
        else:
            self._foreground = Foreground(halo_lensmodel, macromodel_lensmodel, self._z_macro, x_pos, y_pos)

        self._halo_args = halo_args

        self._halo_lensmodel = halo_lensmodel
        self.multi_plane = False
        # this flag needs to be set as False to be compatible with the latest LensEquationSolver feature to make the
        # computation faster

    def ray_shooting(self, x, y, kwargs_lens, check_convention=True):

        if check_convention:
            kwargs_lens = self._set_kwargs(kwargs_lens)

        macromodel_args = []

        for ind in self._macro_indicies:
            macromodel_args.append(kwargs_lens[ind])

        # get the deflection angles from foreground and main lens plane subhalos (once)

        x, y, alphax, alphay = self._foreground.ray_shooting(self._halo_args, macromodel_args, thetax=x, thetay=y,
                                                             force_compute=True)

        x_source, y_source, _, _ = self._full_lensmodel.lens_model.\
            ray_shooting_partial(x, y, alphax, alphay, self._z_macro, self._z_source, kwargs_lens, check_convention=False)

        betax, betay = x_source * self._T_z_source ** -1, y_source * self._T_z_source ** -1

        return betax, betay

    def hessian(self, x, y, kwargs_lens, diff=0.00000001):

        kwargs_lens = self._set_kwargs(kwargs_lens)

        alpha_ra, alpha_dec = self._alpha(x, y, kwargs_lens, check_convention=False)

        alpha_ra_dx, alpha_dec_dx = self._alpha(x + diff, y, kwargs_lens, check_convention=False)
        alpha_ra_dy, alpha_dec_dy = self._alpha(x, y + diff, kwargs_lens, check_convention=False)

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

        kwargs_lens = self._set_kwargs(macromodel_args + self._halo_args)

        x, y, alphax, alphay = self._foreground.ray_shooting(self._halo_args, macromodel_args, offset_index, thetax, thetay,
                                                             force_compute=force_compute)

        x_source, y_source, _, _ = self._full_lensmodel.lens_model.ray_shooting_partial(x, y, alphax, alphay,
                                              self._z_macro, self._z_source, kwargs_lens, check_convention=False)

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

    def _alpha(self, x_pos, y_pos, kwargs_lens, check_convention=True):

        beta_x,beta_y = self.ray_shooting(x_pos, y_pos, kwargs_lens, check_convention)

        alpha_x = np.array(x_pos - beta_x)
        alpha_y = np.array(y_pos - beta_y)

        return alpha_x, alpha_y

    def _set_kwargs(self, kwargs_lens_full):

        if self._observed_convention_index is None:
            return kwargs_lens_full

        kwargs_physical = self._full_lensmodel.lens_model.observed2physical_convention(kwargs_lens_full)

        return kwargs_physical

    def _split_lensmodel(self, lensmodel, lensmodel_args, macro_indicies, numerical_alpha_class,
                         observed_convention_inds):

        """

        :param lensmodel: lensmodel to break up
        :param lensmodel_args: kwargs to break up
        :param z_break: the break redshift
        :param macro_indicies: the indicies of the macromodel in the lens model list
        :return: instances of LensModel for foreground, main lens plane and background halos, and the macromodel
        """

        macro_names, macro_redshifts, macro_args = [], [], []

        halo_names, halo_redshifts, halo_args = [], [], []

        if observed_convention_inds is not None:
            convention_inds = []
        else:
            convention_inds = None

        count = 0

        for i in range(0, len(lensmodel.lens_model_list)):

            z = lensmodel.redshift_list[i]

            if i not in macro_indicies:

                halo_names.append(lensmodel.lens_model_list[i])
                halo_redshifts.append(z)
                halo_args.append(lensmodel_args[i])
                if observed_convention_inds is not None:
                    if i in observed_convention_inds: convention_inds.append(count)
                count += 1

            else:

                macro_names.append(lensmodel.lens_model_list[i])
                macro_redshifts.append(z)
                macro_args.append(lensmodel_args[i])

        macromodel = LensModel(lens_model_list=macro_names, lens_redshift_list=macro_redshifts, cosmo=self._astropy_instance,
                               multi_plane=True,
                               z_source=self._z_source, numerical_alpha_class=numerical_alpha_class)

        halo_lensmodel = LensModel(lens_model_list=halo_names, lens_redshift_list=halo_redshifts,
                                   cosmo=self._astropy_instance, multi_plane=True, z_source=self._z_source,
                                   numerical_alpha_class = numerical_alpha_class, observed_convention_index=convention_inds)

        return macromodel, macro_args, halo_lensmodel, halo_args


class Foreground(object):

    def __init__(self, foreground_lensmodel, macromodel_lensmodel, z_to_vary, x_pos, y_pos, precompupted_rays = None):

        self._halos_lensmodel = foreground_lensmodel
        self._macromodel_lensmodel = macromodel_lensmodel
        self._z_to_vary = z_to_vary
        self._x_pos, self._y_pos = x_pos, y_pos

        dis = self._halos_lensmodel.lens_model._multi_plane_base._cosmo_bkg.T_xy
        if precompupted_rays is None:
            self._rays = [None] * 3
        else:
            self._rays = precompupted_rays

        self._Txy_main = dis(0, z_to_vary)
        z_source = self._halos_lensmodel.lens_model._multi_plane_base._z_source
        self._factor = dis(0, z_source) / dis(z_to_vary, z_source)

    def ray_shooting(self, args, macro_args, offset_index=None, thetax=None, thetay=None, force_compute=True):

        x, y, alphax, alphay = self._ray_shooting_cache(args, offset_index, thetax, thetay, force_compute)

        x, y, alphax, alphay = self._macromodel_lensmodel.lens_model.ray_shooting_partial(x, y, alphax, alphay,
                                  self._z_to_vary, self._z_to_vary, macro_args, include_z_start=True)

        return x, y, alphax, alphay

    def _ray_shooting_cache(self,args,offset_index=None,thetax=None,thetay=None,force_compute=True):

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
