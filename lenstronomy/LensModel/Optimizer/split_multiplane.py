from lenstronomy.LensModel.lens_model import LensModel
import numpy as np

class SplitMultiplane(object):

    z_epsilon = 1e-9

    def __init__(self, x_pos, y_pos, full_lensmodel, lensmodel_params=[], interpolated=False, z_source=None, interp_range=0.001,
                 interp_res = 0.0001, z_macro=None, astropy_instance=None,verbose=False,macro_indicies = None):

        self.interpolated = interpolated

        if self.interpolated and verbose:
            print('interpolation range: '+str(interp_range))
            print('interpolation resolution: '+str(interp_res))
            #print('pixels: '+str(2*interp_range*interp_res**-1)+' pixels per img.')

        self._interp_range = interp_range
        self._interp_steps = 2*interp_range*interp_res**-1

        self.verbose = verbose

        self.z_macro, self.z_source = z_macro, z_source

        self.astropy_instance = astropy_instance

        self.x_pos, self.y_pos = np.array(x_pos), np.array(y_pos)

        self.full_lensmodel, self.lensmodel_params = full_lensmodel, lensmodel_params

        self._T_z_source = full_lensmodel.lens_model._T_z_source

        macromodel_lensmodel, macro_args, halo_lensmodel, halo_args, self._z_background = \
            self._split_lensmodel(full_lensmodel,lensmodel_params,z_break=z_macro,macro_indicies=macro_indicies)

        self.foreground = Foreground(halo_lensmodel,self.z_macro,x_pos,y_pos)
        self.halo_args = halo_args

        self.model_to_vary = ToVary(macromodel_lensmodel,self.z_macro,self._z_background)
        self.macro_args = macro_args

        self.background = Background(halo_lensmodel,self.z_macro,self.z_source,interpolate=interpolated)

    def inherit_rays(self,optimizer_class,plane='foreground'):

        if plane == 'foreground':

            true_rays = optimizer_class.multiplane_optimizer.foreground.rays
            diff_rays = optimizer_class.multiplane_optimizer.foreground.diff_rays
            self.foreground.set_precomputed(true_rays,diff_rays)

    def ray_shooting(self, thetax, thetay, macromodel_args):

        # get the deflection angles from foreground and main lens plane subhalos (once)
        x, y, alphax, alphay = self.foreground.ray_shooting(self.halo_args,thetax=thetax,thetay=thetay)

        x_source, y_source = self._after_foreground(x, y, alphax, alphay, macromodel_args)

        betax, betay = x_source * self._T_z_source ** -1, y_source * self._T_z_source ** -1

        return betax, betay

    def ray_shooting_fast(self, macromodel_args, true_foreground=True, offset_index=None,thetax=None,thetay=None):

        # get the deflection angles from foreground and main lens plane subhalos (once)
        x, y, alphax, alphay = self.foreground.ray_shooting(self.halo_args,true_foreground=true_foreground,
                                                            offset_index=offset_index,thetax=thetax,thetay=thetay
                                                            ,force_compute=False)

        x_source,y_source = self._after_foreground(x,y,alphax,alphay,macromodel_args)

        betax, betay = x_source * self._T_z_source ** -1, y_source * self._T_z_source ** -1

        return betax, betay

    def magnification(self,thetax,thetay,macromodel_args):

        fxx,fxy,fyx,fyy = self.hessian(thetax,thetay,macromodel_args)

        det_J = (1-fxx)*(1-fyy)-fyx*fxy

        return np.absolute(det_J**-1)

    def magnification_fast(self,macromodel_args):

        fxx,fxy,fyx,fyy = self.hessian_fast(macromodel_args)

        det_J = (1-fxx)*(1-fyy)-fyx*fxy

        return np.absolute(det_J**-1)

    def hessian(self,thetax,thetay,macromodel_args,diff=0.00000001):

        alpha_ra, alpha_dec = self._alpha(thetax,thetay, macromodel_args)

        alpha_ra_dx, alpha_dec_dx = self._alpha(thetax + diff, thetay, macromodel_args)
        alpha_ra_dy, alpha_dec_dy = self._alpha(thetax, thetay + diff, macromodel_args)

        dalpha_rara = (alpha_ra_dx - alpha_ra) * diff ** -1
        dalpha_radec = (alpha_ra_dy - alpha_ra) * diff ** -1
        dalpha_decra = (alpha_dec_dx - alpha_dec) * diff ** -1
        dalpha_decdec = (alpha_dec_dy - alpha_dec) * diff ** -1

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra

        return f_xx, f_xy, f_yx, f_yy

    def hessian_fast(self,macromodel_args,diff=0.00000001):

        alpha_ra, alpha_dec = self._alpha_fast(self.x_pos,self.y_pos, macromodel_args, true_foreground=True)

        alpha_ra_dx, alpha_dec_dx = self._alpha_fast(self.x_pos + diff, self.y_pos, macromodel_args, true_foreground=False,
                                                     offset_index=0)
        alpha_ra_dy, alpha_dec_dy = self._alpha_fast(self.x_pos, self.y_pos + diff, macromodel_args, true_foreground=False,
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

    def set_interpolated(self, models, args):

        self.interp_models,self.interp_args = models,args

    def _alpha(self, x_pos, y_pos, macromodel_args):

        beta_x,beta_y = self.ray_shooting(x_pos,y_pos,macromodel_args)

        alpha_x = np.array(x_pos - beta_x)
        alpha_y = np.array(y_pos - beta_y)

        return alpha_x, alpha_y

    def _alpha_fast(self, x_pos, y_pos, macromodel_args, true_foreground=False, offset_index = None):

        beta_x,beta_y = self.ray_shooting_fast(macromodel_args, true_foreground=true_foreground, offset_index=offset_index,
                                               thetax=x_pos,thetay=y_pos)

        alpha_x = np.array(x_pos - beta_x)
        alpha_y = np.array(y_pos - beta_y)

        return alpha_x, alpha_y

    def _after_foreground(self,x,y,alphax,alphay,args):

        x, y, alphax, alphay = self.model_to_vary.ray_shooting(alphax, alphay, args, x, y)

        # compute the angular position on the source plane
        x_source, y_source, _, _ = self.background.ray_shooting(alphax, alphay, self.halo_args, x, y)

        return x_source,y_source

    def _get_interpolated_models(self):

        if not hasattr(self, 'interp_models'):
            return None,None
        else:
            return self.interp_models,self.interp_args

    def _add_no_lens(self,z):

        name = ['SIS']
        args = [{'theta_E': 0., 'center_x': 0, 'center_y': 0}]
        redshift = [z]

        return name,args,redshift

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

        background_z_current = self.z_macro + 0.5*(self.z_source - self.z_macro)

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

        macromodel = LensModel(lens_model_list=macro_names, redshift_list=macro_redshifts, cosmo=self.astropy_instance,
                               multi_plane=True,
                               z_source=self.z_source)

        halo_lensmodel = LensModel(lens_model_list=front_model_names+back_model_names, redshift_list=front_redshifts+back_redshifts,
                                   cosmo=self.astropy_instance,multi_plane=True,z_source=self.z_source)
        halo_args = front_args+back_args

        return macromodel, macro_args, halo_lensmodel, halo_args, background_z_current

class ToVary(object):

    def __init__(self,tovary_lensmodel,z_to_vary,z_next_plane):

        self.tovary_lensmodel = tovary_lensmodel
        self.z_to_vary = z_to_vary
        #self.z_background = z_next_plane

    def ray_shooting(self, thetax, thetay, args, x_in, y_in):

        x, y, alphax, alphay = self.tovary_lensmodel.lens_model. \
            ray_shooting_partial(x_in, y_in, thetax, thetay, z_start=self.z_to_vary,
                                 z_stop=self.z_to_vary, kwargs_lens=args, include_z_start=True)

        return x, y, alphax, alphay

class Foreground(object):

    def __init__(self, foreground_lensmodel, z_to_vary, x_pos, y_pos):

        self.halos_lensmodel = foreground_lensmodel
        self.z_to_vary = z_to_vary
        self.x_pos,self.y_pos = x_pos,y_pos
        self.diff_rays = [None]*2

    def set_precomputed(self,true_rays,offset_rays):

        if true_rays is not None:
            self.rays = true_rays
        if offset_rays is not None:
            self.diff_rays = offset_rays

    def ray_shooting(self,args,true_foreground=False,offset_index=None,thetax=None,thetay=None,force_compute=True):

        if true_foreground:

            if not hasattr(self,'rays'):
                x0, y0 = np.zeros_like(self.x_pos), np.zeros_like(self.y_pos)
                x,y,alphax,alphay = self.halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, self.x_pos, self.y_pos,
                                                                                                 z_start=0,
                                                                                                 z_stop=self.z_to_vary,
                                                                                                 kwargs_lens=args)
                self.rays = {'x':x,'y':y,'alphax':alphax,'alphay':alphay}

            return self.rays['x'],self.rays['y'],self.rays['alphax'],self.rays['alphay']

        elif force_compute:

            x0, y0 = np.zeros_like(self.x_pos), np.zeros_like(self.y_pos)
            x,y,alphax,alphay = self.halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, thetax, thetay,
                                                         z_start=0,z_stop=self.z_to_vary,kwargs_lens=args)
            return x,y,alphax,alphay

        else:

            if self.diff_rays[offset_index] is None:

                x0, y0 = np.zeros_like(self.x_pos), np.zeros_like(self.y_pos)
                x, y, alphax, alphay = self.halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, thetax,
                                                     thetay,z_start=0,z_stop=self.z_to_vary,kwargs_lens=args)

                self.diff_rays[offset_index] = {'x': x, 'y': y, 'alphax': alphax, 'alphay': alphay}

            return self.diff_rays[offset_index]['x'],self.diff_rays[offset_index]['y'],self.diff_rays[offset_index]['alphax'],\
                                self.diff_rays[offset_index]['alphay']

class Background(object):

    def __init__(self, background_lensmodel, z_background, z_source, interpolate, swiveling_cone=False, **kwargs):

        if interpolate is True:

            background_lensmodel,background_args = self._interpolate(background_lensmodel,**kwargs)

        self.halos_lensmodel = background_lensmodel
        self.z_background = z_background
        self.z_source = z_source
        self.interpolate = interpolate
        self.swiveling_cone = swiveling_cone

    def ray_shooting(self, thetax, thetay, args, x_in, y_in):

        if self.interpolate:
            x,y,alphax,alphay = [],[],[],[]
            for i,(thetax_i,thetay_i) in zip(thetax,thetay):
                xi,yi,alphax_i,alphay_i = self.halos_lensmodel[i].lens_model.ray_shooting_partial(x_in[i],
                                              y_in[i],thetax_i,thetay_i,z_start=self.z_background,z_stop=self.z_source,
                                                      kwargs_lens=args)
                x.append(xi)
                y.append(yi)
                alphax.append(alphax_i)
                alphay.append(alphay_i)

            x,y,alphax,alphay = np.array(x),np.array(y),np.array(alphax),np.array(alphay)

        else:

            x, y, alphax, alphay = self.halos_lensmodel.lens_model.ray_shooting_partial(x_in,
                                       y_in, thetax, thetay,z_start=self.z_background,z_stop=self.z_source,kwargs_lens=args)

        return x,y,alphax,alphay

    def _interpolate(self, background_lensmodel, x, y):

        raise Exception('not yet implemented')
        T_z_interp = background_lensmodel.lens_model._T_z_list[0]
        self.interp_models = []
        self.interp_args = []

        x_values, y_values = np.linspace(-self._interp_range, self._interp_range, self._interp_steps), \
                             np.linspace(-self._interp_range, self._interp_range, self._interp_steps)

        for count, (xi, yi) in enumerate(zip(x, y)):

            if self.verbose:
                print('interpolating field behind image ' + str(count + 1) + '...')

            interp_model_i, interp_args_i = self._lensmodel_interpolated((x_values + xi) * T_z_interp ** -1,
                                                                         (y_values + yi) * T_z_interp ** -1,
                                                                         self.background_lensmodel,
                                                                         self.background_args)

            self.interp_models.append(interp_model_i)
            self.interp_args.append(interp_args_i)

        return self.interp_models, self.interp_args

    def _lensmodel_interpolated(self, x_values, y_values, interp_lensmodel, interp_args):

        """

        :param x_values: 1d array of x coordinates to interpolate
        :param y_values: 1d array of y coordinates to interpolate
        (e.g. np.linspace(ymin,ymax,steps))
        :param interp_lensmodel: lensmodel to interpolate
        :param interp_args: kwargs for interp_lensmodel
        :return: interpolated lensmodel
        """
        xx, yy = np.meshgrid(x_values, y_values)
        L = int(len(x_values))
        xx, yy = xx.ravel(), yy.ravel()

        f_x, f_y = interp_lensmodel.alpha(xx, yy, interp_args)

        interp_args = [{'f_x': f_x.reshape(L, L), 'f_y': f_y.reshape(L, L),
                        'grid_interp_x': x_values, 'grid_interp_y': y_values}]

        return LensModel(lens_model_list=['INTERPOL'], redshift_list=[self._z_background], cosmo=self.astropy_instance,
                         z_source=self.z_source, multi_plane=True), interp_args


