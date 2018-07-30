from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel
import numpy as np

class MultiPlaneOptimizer(object):

    def __init__(self, lensmodel_full, all_args, x_pos, y_pos, tol_source, Params, magnification_target,
                 tol_mag, centroid_0, tol_centroid, z_main, z_src, astropy_instance, interpolated,return_mode = 'PSO',
                 mag_penalty=False,return_array = False, verbose=False):

        self.Params = Params
        self.lensModel = lensmodel_full
        self.all_lensmodel_args = all_args
        self.solver = LensEquationSolver(self.lensModel)

        self.tol_source = tol_source

        self.magnification_target = magnification_target
        self.tol_mag = tol_mag

        self._compute_mags = mag_penalty

        self._return_array = return_array

        self.centroid_0 = centroid_0
        self.tol_centroid = tol_centroid

        self._counter = 1
        self.verbose = verbose

        self._x_pos,self._y_pos = x_pos,y_pos
        self.mag_penalty,self.src_penalty,self.parameters = [],[], []

        self.multiplane_optimizer = SplitMultiplane(x_pos,y_pos,lensmodel_full,all_args,interpolated=interpolated,
                                                    z_source=z_src,z_macro=z_main,astropy_instance=astropy_instance)

        self._converged = False
        self._return_mode = return_mode

    def _init_particles(self,n_particles,n_iterations):

        self._n_total_iter = n_iterations*n_particles
        self._n_particles = n_particles
        self._mag_penalty_switch = 1

    def _get_images(self,args):

        srcx, srcy = self.lensModel.ray_shooting(self._x_pos,self._y_pos,args)

        self.source_x, self.source_y = np.mean(srcx), np.mean(srcy)
        x_image, y_image = self.solver.image_position_from_source(self.source_x, self.source_y, args, precision_limit=10**-10)

        return x_image, y_image

    def _source_position_penalty(self, lens_args_tovary):

        betax,betay = self.multiplane_optimizer.ray_shooting(lens_args_tovary)
        self._betax,self._betay = betax,betay

        dx = ((betax[0] - betax[1]) ** 2 + (betax[0] - betax[2]) ** 2 + (betax[0] - betax[3]) ** 2 + (
                betax[1] - betax[2]) ** 2 +
              (betax[1] - betax[3]) ** 2 + (betax[2] - betax[3]) ** 2)
        dy = ((betay[0] - betay[1]) ** 2 + (betay[0] - betay[2]) ** 2 + (betay[0] - betay[3]) ** 2 + (
                betay[1] - betay[2]) ** 2 +
              (betay[1] - betay[3]) ** 2 + (betay[2] - betay[3]) ** 2)

        if self._return_array:
            return 0.5 * np.array([dx,dy]) * self.tol_source ** -2
        else:
            return 0.5 * (dx + dy) * self.tol_source ** -2


    def _magnification_penalty(self, lens_args, magnification_target, tol):

        magnifications = self.multiplane_optimizer.magnification(lens_args,split_jacobian=False)

        magnifications *= max(magnifications) ** -1

        dM = []

        for i, target in enumerate(magnification_target):
            mag_tol = tol * target
            dM.append((magnifications[i] - target) * mag_tol ** -1)

        dM = np.array(dM)

        if self._return_array:
            return 0.5*dM**2
        else:
            return 0.5 * np.sum(dM ** 2)

    def _centroid_penalty(self, values_dic, tol_centroid):

        d_centroid = ((values_dic[0]['center_x'] - self.centroid_0[0]) * tol_centroid ** -1) ** 2 + \
                     ((values_dic[0]['center_y'] - self.centroid_0[1]) * tol_centroid ** -1) ** 2

        return 0.5 * d_centroid

    def _log(self,src_penalty,mag_penalty):

        if mag_penalty is None:
            mag_penalty = np.inf
        if src_penalty is None:
            src_penalty = np.inf

        self.src_penalty.append(np.sum(src_penalty))
        self.mag_penalty.append(np.sum(mag_penalty))
        self.parameters.append(self.lens_args_latest)

    def _compute_mags_criterion(self):

        if self._compute_mags:
            return True

        if self._counter > self._n_particles and np.mean(self.src_penalty[-self._n_particles:]) < 1:
            return True
        else:
            return False

    def __call__(self, lens_values_tovary, src_penalty=None,mag_penalty=None,centroid_penalty=None):

        self._counter += 1

        params_fixed = self.Params.argsfixed_todictionary()
        lens_args_tovary = self.Params.argstovary_todictionary(lens_values_tovary)

        if self.tol_source is not None:

            src_penalty = self._source_position_penalty(lens_args_tovary)

            self._compute_mags = self._compute_mags_criterion()

        if self._compute_mags and self.tol_mag is not None:
            mag_penalty = self._magnification_penalty(lens_args_tovary+params_fixed,self.magnification_target,self.tol_mag)

        if self.tol_centroid is not None:
            centroid_penalty = self._centroid_penalty(lens_args_tovary,self.tol_centroid)

        if self._return_array:
            penalty = src_penalty

            if self._compute_mags and self.tol_mag is not None:
                penalty = np.append(penalty,mag_penalty)

            if self.tol_centroid is not None:
                penalty = np.append(penalty,centroid_penalty)

        else:
            _penalty = [src_penalty,mag_penalty,centroid_penalty]

            penalty = 0
            for pen in _penalty:
                if pen is not None:
                    penalty += pen

        if self._counter % 500 == 0 and self.verbose:

            print('source penalty: '), src_penalty
            if self.mag_penalty is not None:
                print('mag penalty: '), mag_penalty

        self.lens_args_latest = lens_args_tovary + params_fixed

        self._log(src_penalty,mag_penalty)

        if self._return_mode == 'PSO':
            return -1 * penalty, None
        else:

            return np.array(penalty)

class SplitMultiplane(object):

    z_epsilon = 1e-16

    def __init__(self, x_pos, y_pos, full_lensmodel, lensmodel_params=[], interpolated=False, z_source=None, interp_range=0.1,
                 interp_res = 0.0001, z_macro=None, astropy_instance=None):

        self.interpolated = interpolated
        self._interp_range = interp_range
        self._interp_steps = 2*interp_range*interp_res**-1

        self.z_macro, self.z_source = z_macro, z_source
        self.astropy_instance = astropy_instance

        self.x_pos, self.y_pos = x_pos, y_pos

        self.full_lensmodel, self.lensmodel_params = full_lensmodel, lensmodel_params

        self._z_background = self._background_z(full_lensmodel, z_macro)

        self._T_z_source = full_lensmodel.lens_model._T_z_source

        self.macromodel_lensmodel, _, back_lensmodel, back_args, self.halos_lensmodel, self.halos_args\
            = self._split_lensmodel(full_lensmodel,lensmodel_params,z_break=z_macro)

        self.background_lensmodel, self.background_args = back_lensmodel, back_args

    def magnification(self, args, split_jacobian=False):

        if split_jacobian:

            raise Exception('not yet implemented.')

        else:

            magnification = self.full_lensmodel.magnification(self.x_pos,self.y_pos,args)

        return np.absolute(magnification)

    def ray_shooting(self, macromodel_args):

        # get the deflection angles from foreground and main lens plane subhalos (once)
        x, y, alphax, alphay = self._foreground_deflections()

        # add the deflections from the macromodel
        x,y,alphax,alphay = self.macromodel_lensmodel.lens_model.ray_shooting_partial(x, y, alphax,
                                      alphay, self.z_macro - self.z_epsilon, self._z_background, macromodel_args)

        x_source, y_source, _, _ = self._background_deflections(x,y, alphax,alphay,self.interpolated)

        # compute the angular position on the source plane
        betax, betay = x_source * self._T_z_source ** -1, y_source * self._T_z_source ** -1

        return betax, betay

    def _background_deflections(self,x,y,alphax,alphay,interpolated):

        if interpolated is False:
            x,y,alphax,alphay = self.background_lensmodel.lens_model.ray_shooting_partial(x,y,alphax,alphay,self.z_macro,
                                                                               self.z_source,self.background_args)

        else:

            if not hasattr(self,'interp_models'):

                T_z_interp = self.background_lensmodel.lens_model._T_z_list[0]
                self.interp_models = []
                self.interp_args = []

                x_values, y_values = np.linspace(-self._interp_range,self._interp_range,self._interp_steps),\
                                     np.linspace(-self._interp_range,self._interp_range,self._interp_steps)

                count = 1
                for xi,yi in zip(x,y):

                    print('interpolating field behind image '+str(count)+'...')
                    count+=1
                    interp_model_i,interp_args_i = self._lensmodel_interpolated((x_values+xi)*T_z_interp**-1,
                                                        (y_values+yi)*T_z_interp**-1, self.background_lensmodel,self.background_args)

                    self.interp_models.append(interp_model_i)
                    self.interp_args.append(interp_args_i)

            x_out,y_out,alphax_out,alphay_out = [],[],[],[]

            for i in range(0,len(x)):
                xi,yi,alphax_i,alphay_i = self.interp_models[i].lens_model.ray_shooting_partial(x[i],y[i],alphax[i],
                                                               alphay[i],self.z_macro,self.z_source,self.interp_args[i])
                x_out.append(xi)
                y_out.append(yi)
                alphax_out.append(alphax_i)
                alphay_out.append(alphay_i)

            x,y,alphax,alphay = np.array(x_out),np.array(y_out),np.array(alphax_out),np.array(alphay_out)

        return x,y,alphax,alphay

    def _foreground_deflections(self):

        """

        :param x_pos: observed x position
        :param y_pos: observed y position

        :return: foreground deflections
        """

        if not hasattr(self, 'alphax_foreground'):

            x0, y0 = np.zeros_like(self.x_pos), np.zeros_like(self.y_pos)

            # ray shoot through the halos in front and in the main lens plane
            self.x_macro, self.y_macro, self.alphax_foreground, self.alphay_foreground = \
                self.halos_lensmodel.lens_model.ray_shooting_partial(x0, y0, self.x_pos, self.y_pos, z_start=0,
                                                          z_stop=self.z_macro,
                                                          kwargs_lens=self.halos_args)


        return self.x_macro, self.y_macro, self.alphax_foreground, self.alphay_foreground

    def _add_no_lens(self,z):

        name = ['SIS']
        args = [{'theta_E': 0., 'center_x': 0, 'center_y': 0}]
        redshift = [z]

        return name,args,redshift

    def _split_lensmodel(self, lensmodel, lensmodel_args, z_break, macro_indicies=[0, 1]):

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
        main_halo_names, main_halo_redshifts, main_halo_args = [], [], []

        halo_names, halo_redshifts, halo_args = [], [], []

        for i in range(0, len(lensmodel.lens_model_list)):

            if i not in macro_indicies:

                halo_names.append(lensmodel.lens_model_list[i])
                halo_redshifts.append(lensmodel.redshift_list[i])
                halo_args.append(lensmodel_args[i])

                if lensmodel.redshift_list[i] > z_break:
                    back_model_names.append(lensmodel.lens_model_list[i])
                    back_redshifts.append(lensmodel.redshift_list[i])
                    back_args.append(lensmodel_args[i])
                elif lensmodel.redshift_list[i] < z_break:
                    front_model_names.append(lensmodel.lens_model_list[i])
                    front_redshifts.append(lensmodel.redshift_list[i])
                    front_args.append(lensmodel_args[i])
                else:
                    main_halo_names.append(lensmodel.lens_model_list[i])
                    main_halo_redshifts.append(z_break)
                    main_halo_args.append(lensmodel_args[i])

            else:

                macro_names.append(lensmodel.lens_model_list[i])
                macro_redshifts.append(lensmodel.redshift_list[i])
                macro_args.append(lensmodel_args[i])

        macromodel = LensModel(lens_model_list=macro_names, redshift_list=macro_redshifts, cosmo=self.astropy_instance,
                               multi_plane=True,
                               z_source=self.z_source)

        if len(front_model_names) == 0:
            front_model_names,front_args,front_redshifts = self._add_no_lens(self.z_macro*0.5)

        front_halos = LensModel(lens_model_list=front_model_names, redshift_list=front_redshifts,
                                cosmo=self.astropy_instance, multi_plane=True,
                                z_source=self.z_source)

        if len(back_model_names) == 0:
            f = 0.1*(self.z_source - self.z_macro)*self.z_macro**-1
            back_model_names,back_args,back_redshifts = self._add_no_lens(self.z_macro*(1+f))

        back_halos = LensModel(lens_model_list=back_model_names, redshift_list=back_redshifts,
                               cosmo=self.astropy_instance, multi_plane=True,
                               z_source=self.z_source)

        if len(main_halo_names) == 0:
            main_halo_names,main_halo_args,main_halo_redshifts = self._add_no_lens(self.z_macro*0.5)

        main_halos = LensModel(lens_model_list=main_halo_names, redshift_list=main_halo_redshifts,
                               cosmo=self.astropy_instance,
                               z_source=self.z_source, multi_plane=True)

        if len(halo_names) == 0:
            halo_names,halo_args,halo_redshifts = self._add_no_lens(self.z_macro)
        halos = LensModel(lens_model_list=halo_names, redshift_list=halo_redshifts, cosmo=self.astropy_instance,
                          z_source=self.z_source,
                          multi_plane=True)

        return macromodel, macro_args, back_halos, back_args, halos, halo_args

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

    def _background_z(self, lensModel, z_macro):

        # computes the redshift of the first lens plane behind the main lens

        for i in lensModel.lens_model._sorted_redshift_index:

            if lensModel.redshift_list[i] > z_macro:
                return lensModel.redshift_list[i]

        return z_macro
