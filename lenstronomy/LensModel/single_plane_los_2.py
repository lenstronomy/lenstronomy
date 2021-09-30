__author__ = 'nataliehogg', 'pierrefleury'

#import numpy as np
from lenstronomy.LensModel.single_plane import SinglePlane

__all__ = ['SinglePlaneLOS']


class SinglePlaneLOS(SinglePlane):
    """
    This class is based on the 'SinglePlane' class, modified to include
    line-of-sight effects as presented by Fleury et al. in 2104.08883.
    
    Are modified:
    - init (to include a new attribute, self.los)
    - alpha
    - hessian
    
    Are unchanged (inherited from SinglePlane):
    - ray_shooting, because it calls the modified alpha
    - mass_2d, mass_3d, density which refer to the main lens without LOS
    corrections.
    
    To be done: implementation of the time delays, so that it can be used
    by LensModel.
    """
    
    
    def __init__(self, lens_model_list,
                 numerical_alpha_class=None,
                 lens_redshift_list=None, 
                 z_source_convention=None,
                 kwargs_interp=None):
        """
        Instance of SinglePlaneLOS() based on the SinglePlane() but
        to which we add, as an attribute, the line-of-sight class
        extracted from the lens_model_list.
        """
        
        # Extract LOS from lens_model_list
        try:
            self.index_los = lens_model_list.index('LOS')
        except ValueError:
            print("""
                  You tried to do single-plane lensing with line-of-sight
                  effect, but you did not include 'LOS' in the list of lens
                  models.
                  """)
        self.los = self._import_class('LOS', custom_class=None,
                                      kwargs_interp=None)
        
        # Proceed with the rest of the lenses
        lens_model_list_wo_los = [
            model for i, model in enumerate(lens_model_list)
            if i != self.index_los]
        super().__init__(lens_model_list_wo_los)
        
        
    def split_lens_los(self, kwargs):
        """
        This function splits the list of key-word arguments given to the lens
        model into those that correspond to the lens itself (kwargs_lens), and
        those that correspond to the line-of-sight corrections (kwargs_los).
        """
        
        kwargs_los = kwargs[self.index_los]
        kwargs_lens = [kwarg for i, kwarg in enumerate(kwargs)
                       if i != self.index_los]
        
        return kwargs_lens, kwargs_los
        
        
    def alpha(self, x, y, kwargs, k=None):
        """
        Displacement angle including the line-of-sight corrections
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters
        matching the lens model classes, including line-of-sight corrections
        :param k: only evaluate the k-th lens model
        :return: deflection angles in units of arcsec
        """
        
        kwargs_lens, kwargs_los = self.split_lens_los(kwargs)
        
        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])
        
        # Displacement due to the main lens only
        f_x, f_y = super().alpha(x_d, y_d, kwargs=kwargs_lens, k=k)
        
        # Correction due to the background convergence and shear
        f_x, f_y = self.los.distort_vector(f_x, f_y,
                                           kappa=kwargs_los['kappa_ds'],
                                           gamma1=kwargs_los['gamma1_ds'],
                                           gamma2=kwargs_los['gamma2_ds'])
        
        # Sheared position in the absence of the main lens
        x_os, y_os =  self.los.distort_vector(x, y,
                                              kappa=kwargs_los['kappa_os'],
                                              gamma1=kwargs_los['gamma1_os'],
                                              gamma2=kwargs_los['gamma2_os'])

        # Complete displacement
        f_x += x - x_os
        f_y += y - y_os
        
        return f_x, f_y


    def hessian(self, x, y, kwargs, k=None):
        """
        Hessian matrix
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: f_xx, f_xy, f_yx, f_yy components
        """
        
        kwargs_lens, kwargs_los = self.split_lens_los(kwargs)
        
        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])
        
        # Hessian matrix of the main lens only
        f_xx, f_xy, f_yx, f_yy = super().hessian(x_d, y_d,
                                                 kwargs=kwargs_lens, k=k)
        
        # Multiply on the left by (1 - Gamma_ds)
        f_xx, f_xy, f_yx, f_yy = self.los.left_multiply(
                                    f_xx, f_xy, f_yx, f_yy,
                                    kappa=kwargs_los['kappa_ds'],
                                    gamma1=kwargs_los['gamma1_ds'],
                                    gamma2=kwargs_los['gamma2_ds'])
        
        # Multiply on the right by (1 - Gamma_od)
        f_xx, f_xy, f_yx, f_yy = self.los.right_multiply(
                                    f_xx, f_xy, f_yx, f_yy,
                                    kappa=kwargs_los['kappa_od'],
                                    gamma1=kwargs_los['gamma1_od'],
                                    gamma2=kwargs_los['gamma2_od'])
        
        # LOS contribution in the absence of the main lens
        f_xx += kwargs_los['kappa_os'] + kwargs_los['gamma1_os']
        f_xy += kwargs_los['gamma2_os']
        f_yx += kwargs_los['gamma2_os']
        f_yy += kwargs_los['kappa_os'] - kwargs_los['gamma1_os']
        
        return f_xx, f_xy, f_yx, f_yy

