from lenstronomy.LensModel.QuadOptimizer.multi_scale_model import MultiScaleModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.lens_model import LensModel
from multiprocessing.pool import Pool
import numpy as np

# def compute_magnifications_pool(args_list, nproc):
#
#     pool = Pool(nproc)
#     out = pool.map(compute_magnifications, args_list)
#     pool.close()
#     return out

def compute_magnifications_pool(args_list, pool):

    out = pool.map(compute_magnifications, args_list)

    return out

def compute_magnifications(args):
    
    fit_setting = 'FREE'
    kwargs_magnification_finite = {'grid_resolution': 0.001}

    (sample_index, matching_scale, source_fwhm_pc, z_lens, z_source, kwargs_mcmc_fit,
     lens_model_fit, kwargs_lens_fit, lens_model_list_other, redshift_list_other, kwargs_lens_other,
     fit_model) = args

    kwargs_lens_fit = kwargs_mcmc_fit['kwargs_lens']
    kwargs_ps_fit = kwargs_mcmc_fit['kwargs_ps'][0]

    x_img, y_img = kwargs_ps_fit['ra_image'], kwargs_ps_fit['dec_image']
    source_x_sample, source_y_sample = lens_model_fit.ray_shooting(x_img, y_img, kwargs_lens_fit)
    src_x, src_y = np.mean(source_x_sample), np.mean(source_y_sample)

    local_image_model = LocalImageModel(x_img, y_img,
                                        src_x, src_y,
                                        lens_model_list_other, list(redshift_list_other),
                                        kwargs_lens_other, z_lens, z_source)

    fxx_fit, fxy_fit, fyx_fit, fyy_fit = lens_model_fit.hessian(x_img, y_img, kwargs_lens_fit,
                                                                diff=matching_scale)
    kappa_constraint = 0.5 * (fxx_fit + fyy_fit)
    gamma1_constraint = 0.5 * (fxx_fit - fyy_fit)
    gamma2_constraint = 0.5 * (fxy_fit + fyx_fit)

    m = local_image_model.model_magnifications(source_fwhm_pc, kappa_constraint, gamma1_constraint,
                                            gamma2_constraint, matching_scale, fit_setting, curved_arc_init=None,
                                            lens_model_init=lens_model_fit, kwargs_lens_init=kwargs_lens_fit,
                                            verbose=False, fit_model=fit_model,
                                            **kwargs_magnification_finite)
    return m

class LocalImageModel(object):

    def __init__(self, x_image_coordinate, y_image_coordinate, source_x_coordinate, source_y_coordinate,
                 lens_model_list_other, redshift_list_other,
                 kwargs_lens_other, z_lens, z_source, astropy_instance=None):

        self._image_models = []
        self._nimg = len(x_image_coordinate)

        for (xcoord, ycoord) in zip(x_image_coordinate, y_image_coordinate):

            new = MultiScaleModel(xcoord, ycoord, source_x_coordinate, source_y_coordinate,
                   lens_model_list_other, redshift_list_other, kwargs_lens_other,
                                  z_lens, z_source, astropy_instance)

            self._image_models.append(new)

        self._source_x_coordinate = source_x_coordinate
        self._source_y_coordinate = source_y_coordinate
        self._lens_model_list_other = lens_model_list_other
        self._redshift_list_other = redshift_list_other
        self._kwargs_lens_other = kwargs_lens_other
        self._x_image_coords = x_image_coordinate
        self._y_image_coords = y_image_coordinate
        self._zlens = z_lens
        self._zsource = z_source
        self._cosmo = astropy_instance

    def estimate_curved_arc(self, lens_model, kwargs_lens):

        ext = LensModelExtensions(lens_model)
        estimate = []
        for i in range(0, len(self._x_image_coords)):
            estimate.append(ext.curved_arc_estimate(self._x_image_coords[i], self._y_image_coords[i], kwargs_lens))
        return estimate

    def shift_lensmodels(self):

        kwargs_shift_list = []
        for model in self._image_models:

            kwargs_shift = model.compute_kwargs_shift()
            kwargs_shift_list.append(kwargs_shift)

        lensmodel_shift = LensModel(['SHIFT'] + self._lens_model_list_other,
                                    lens_redshift_list=[self._zlens] + list(self._redshift_list_other),
                                    z_source=self._zsource, multi_plane=True, cosmo=self._cosmo)
        return lensmodel_shift, kwargs_shift_list

    def model_hessian(self, kappa_constraint_list, gamma1_constraint_list, gamma2_constraint_list,
                         angular_matching_scale, hessian_init=None,
                         lens_model_init=None, kwargs_lens_init=None, verbose=False):

        if len(kappa_constraint_list) != len(gamma1_constraint_list) or len(gamma1_constraint_list) != len(gamma2_constraint_list):
            raise Exception('kappa_constraint_list, gamma1_constraint_list, and gamma2_constraint_list '
                            'must all be the same length.')
        if len(kappa_constraint_list) != self._nimg:
            raise Exception('kappa_cosntraint_list, gamma1_constraint_list, and gamma2_constraint_list constraints must all have len == number of images')

        if hessian_init is None:
            if lens_model_init is None or kwargs_lens_init is None:
                raise Exception('If hessian_init is not specified, must provide an instance of LensModel and '
                                'a kwargs_lens dict with which to estimate the hessian properties')

        hessian_init = []
        for i in range(0, self._nimg):
            fxx, fxy, fyx, fyy = lens_model_init.hessian(self._x_image_coords[i],
                                                         self._y_image_coords[i],
                                                         kwargs_lens_init,
                                                         diff=angular_matching_scale)
            hessian_init.append({'f_xx': fxx, 'f_xy': fxy, 'f_yx': fyx, 'f_yy': fyy})

        kwargs_hessian = []
        kwargs_full = []
        result = [None] * len(self._image_models)

        for i, model in enumerate(self._image_models):
            if verbose:
                print('solving for image '+str(i+1)+' with constraints: ')
                print('kappa = '+str(kappa_constraint_list[i]))
                print('gamma1 = ' + str(gamma1_constraint_list[i]))
                print('gamma2 = ' + str(gamma2_constraint_list[i]))

            kw_hess, kw_full, r = model.solve_kwargs_hessian(hessian_init[i],
                                                        kappa_constraint_list[i],
                                                        gamma1_constraint_list[i],
                                                        gamma2_constraint_list[i],
                                                        angular_matching_scale)

            kwargs_hessian.append(kw_hess)
            kwargs_full.append(kw_full)
            result[i] = r
            if verbose: print('result: ', r)

        lens_model_hessian = LensModel(['HESSIAN', 'SHIFT'] + self._lens_model_list_other,
                                          z_lens=self._zlens, z_source=self._zsource,
                                          lens_redshift_list=[self._zlens] * 2 + list(self._redshift_list_other),
                                          cosmo=None, multi_plane=True)

        return kwargs_hessian, kwargs_full, lens_model_hessian, result

    def model_curved_arc(self, kappa_constraint_list, gamma1_constraint_list, gamma2_constraint_list,
                         angular_matching_scale, fit_setting, curved_arc_init=None,
                         lens_model_init=None, kwargs_lens_init=None, verbose=False):

        if len(kappa_constraint_list) != len(gamma1_constraint_list) or len(gamma1_constraint_list) != len(gamma2_constraint_list):
            raise Exception('kappa_constraint_list, gamma1_constraint_list, and gamma2_constraint_list '
                            'must all be the same length.')
        if len(kappa_constraint_list) != self._nimg:
            raise Exception('kappa_cosntraint_list, gamma1_constraint_list, and gamma2_constraint_list constraints must all have len == number of images')

        if curved_arc_init is None:
            if lens_model_init is None or kwargs_lens_init is None:
                raise Exception('If curved_arc_init is not specified, must provide an instance of LensModel and '
                                'a kwargs_lens dict with which to estimate the curved arc properties')
            curved_arc_init = self.estimate_curved_arc(lens_model_init, kwargs_lens_init)

        if len(curved_arc_init) != self._nimg:
            raise Exception('length of curved_arc_init must equal number of images')

        kwargs_curved_arc = []
        kwargs_full = []
        result = [None] * len(self._image_models)

        for i, model in enumerate(self._image_models):
            if verbose:
                print('solving for image '+str(i+1)+' with constraints: ')
                print('kappa = '+str(kappa_constraint_list[i]))
                print('gamma1 = ' + str(gamma1_constraint_list[i]))
                print('gamma2 = ' + str(gamma2_constraint_list[i]))

            kw_arc, kw_full, r = model.solve_kwargs_arc(curved_arc_init[i],
                                                        kappa_constraint_list[i],
                                                        gamma1_constraint_list[i],
                                                        gamma2_constraint_list[i],
                                                        angular_matching_scale,
                                                        fit_setting)

            kwargs_curved_arc.append(kw_arc)
            kwargs_full.append(kw_full)
            result[i] = r
            if verbose: print('result: ', r)

        lens_model_curved_arc = LensModel(['CURVED_ARC', 'SHIFT'] + self._lens_model_list_other,
                                          z_lens=self._zlens, z_source=self._zsource,
                                          lens_redshift_list=[self._zlens] * 2 + list(self._redshift_list_other),
                                          cosmo=None, multi_plane=True)

        return kwargs_curved_arc, kwargs_full, lens_model_curved_arc, result

    def model_magnifications(self, source_fwhm_parsec, kappa_constraint_list, gamma1_constraint_list,
                         gamma2_constraint_list, angular_matching_scale, fit_setting, curved_arc_init=None,
                         lens_model_init=None, kwargs_lens_init=None, verbose=False, fit_model='CURVED_ARC',
                         **kwargs_magnification_finite):

        if fit_model == 'CURVED_ARC':
            _, kwargs_lens, lens_model, _ = self.model_curved_arc(kappa_constraint_list, gamma1_constraint_list, gamma2_constraint_list,
                         angular_matching_scale, fit_setting, curved_arc_init,
                         lens_model_init, kwargs_lens_init, verbose)
        elif fit_model == 'HESSIAN':
            _, kwargs_lens, lens_model, _ = self.model_hessian(kappa_constraint_list, gamma1_constraint_list, gamma2_constraint_list,
                         angular_matching_scale, curved_arc_init,
                         lens_model_init, kwargs_lens_init, verbose)
        else:
            raise Exception('fit_model must be HESSIAN or CURVED_ARC')

        extension = LensModelExtensions(lens_model)

        magnifications = []
        for i in range(0, len(self._x_image_coords)):
            mag = extension.magnification_finite_adaptive([self._x_image_coords[i]],
                                                                 [self._y_image_coords[i]],
                                                                     self._source_x_coordinate,
                                                                     self._source_y_coordinate,
                                                                     kwargs_lens[i], source_fwhm_parsec,
                                                          self._zsource, **kwargs_magnification_finite)
            magnifications.append(float(mag))

        return np.array(magnifications)


