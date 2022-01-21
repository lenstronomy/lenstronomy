__author__ = 'nataliehogg', 'pierrefleury', 'danjohnson98'

from lenstronomy.LensModel.single_plane import SinglePlane
#import numpy as np
import copy

__all__ = ['SinglePlaneLOS']


class SinglePlaneLOS(SinglePlane):
    """
    This class is based on the 'SinglePlane' class, modified to include
    line-of-sight effects as presented by Fleury et al. in 2104.08883.

    Are added:                          #DJMod
    - effective potential
    - effective alpha
    
    Are modified:
    - init (to include a new attribute, self.los)
    - fermat potential                 #DJMod
    - alpha
    - hessian

    Are unchanged (inherited from SinglePlane):
    - ray_shooting, because it calls the modified alpha
    - mass_2d, mass_3d, density which refer to the main lens without LOS
    corrections.
    """


    def __init__(self, lens_model_list, index_los,
                 numerical_alpha_class=None,
                 lens_redshift_list=None,
                 z_source_convention=None,
                 kwargs_interp=None):
        """
        Instance of SinglePlaneLOS() based on the SinglePlane(), except:
        - argument "index_los" indicating the position of the LOS model in the
        lens_model_list (for correct association with kwargs)
        - attribute "los" containing the LOS model.
        """
        
        # Extract the los model
        self.index_los = index_los
        self.los_model = lens_model_list[index_los]
        self.los = self._import_class(self.los_model, custom_class=None, kwargs_interp=None)

        # Proceed with the rest of the lenses
        lens_model_list_wo_los = [
            model for i, model in enumerate(lens_model_list)
            if i != index_los]
        super().__init__(lens_model_list_wo_los)


    def split_lens_los(self, kwargs):
        """
        This function splits the list of key-word arguments given to the lens
        model into those that correspond to the lens itself (kwargs_lens), and
        those that correspond to the line-of-sight corrections (kwargs_los).
        """

        kwargs_los = copy.deepcopy(kwargs[self.index_los])
        # if 'LOS_MINIMAL' is at play, we set Gamma_os = Gamma_los
        # and Gamma_ds = Gamma_od
        if self.los_model == 'LOS_MINIMAL':
            kwargs_los['kappa_os'] = kwargs_los.pop('kappa_los')
            kwargs_los['gamma1_os'] = kwargs_los.pop('gamma1_los')
            kwargs_los['gamma2_os'] = kwargs_los.pop('gamma2_los')
            kwargs_los['omega_os'] = kwargs_los.pop('omega_los')
            kwargs_los['kappa_ds'] = kwargs_los['kappa_od']
            kwargs_los['gamma1_ds'] = kwargs_los['gamma1_od']
            kwargs_los['gamma2_ds'] = kwargs_los['gamma2_od']
            kwargs_los['omega_ds'] = kwargs_los['omega_od']
                
        kwargs_lens = [kwarg for i, kwarg in enumerate(kwargs)
                       if i != self.index_los]

        return kwargs_lens, kwargs_los

    def effective_potential(self, x, y, kwargs, k=None):                    #DJMod
        """
        the lensing potential of the main lens, evaluated at a position which is modified by LOS effects
        between the observer and the lens(equation 3.14 in Fleury et al.)
        
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing potential in units of arcsec^2
        """

        kwargs_lens, kwargs_los = self.split_lens_los(kwargs)
        
        # Angular position where the ray hits the deflector's plane 
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        #Evaluating the potential of the main lens at this position
        effective_potential = super().potential(x_d, y_d, kwargs=kwargs_lens, k=k)
        
        return effective_potential

    def effective_alpha(self, x, y, kwargs, k=None):                    #DJMod
        """
        the effective displacement angle (equation 3.15 in Fleury et al.) 
        
        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: effective deflection angles in units of arcsec
        """

        kwargs_lens, kwargs_los = self.split_lens_los(kwargs)

        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        #the displacement due to the main lens only is then evaluated at this position
        f_x, f_y = super().alpha(x_d, y_d, kwargs=kwargs_lens, k=k)

        #these angles are then further modified as follows
        f_x_d, f_y_d = self.los.distort_vector(f_x, f_y,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        #f_x_d and f_y_d are the components of alpha_eff
        
        return f_x_d, f_y_d


    def fermat_potential(self, x_image, y_image, kwargs, x_source=None, y_source=None, k=None):         #DJMod
        """
        Calculates the Fermat Potential with LOS corrections in the tidal regime (equation 3.12 in Fleury et. al.)

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 without geometry term (second part of Eqn 1 in Suyu et al. 2013) as a list
        """

        kwargs_lens, kwargs_los = self.split_lens_los(kwargs)

        #calculating the effective potential
        effective_potential = self.effective_potential(x_image, y_image, kwargs, k=k) 

        #the effective displacement angles
        f_x, f_y = self.effective_alpha(x_image, y_image, kwargs, k=k) 

        #deflection angle distorted by \Gamma_LOS matrix
        a_x, a_y = self.los.distort_vector(f_x, f_y,
                                           kappa=-kwargs_los['kappa_od']-kwargs_los['kappa_os']+kwargs_los['kappa_ds'],
                                           omega=-kwargs_los['omega_od']-kwargs_los['omega_os']+kwargs_los['omega_ds'],
                                           gamma1=-kwargs_los['gamma1_od']-kwargs_los['gamma1_os']+kwargs_los['gamma1_ds'],
                                           gamma2=-kwargs_los['gamma2_od']-kwargs_los['gamma2_os']+kwargs_los['gamma2_ds'])

        #we can then obtain the geometrical term
        geometry = (f_x*a_x + f_y*a_y) / 2.
        
        return geometry - effective_potential

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
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        # Displacement due to the main lens only
        f_x, f_y = super().alpha(x_d, y_d, kwargs=kwargs_lens, k=k)

        # Correction due to the background convergence, shear and rotation
        f_x, f_y = self.los.distort_vector(f_x, f_y,
                                           kappa=kwargs_los['kappa_ds'],
                                           omega=kwargs_los['omega_ds'],
                                           gamma1=kwargs_los['gamma1_ds'],
                                           gamma2=kwargs_los['gamma2_ds'])

        # Perturbed position in the absence of the main lens
        x_os, y_os = self.los.distort_vector(x, y,
                                             kappa=kwargs_los['kappa_os'],
                                             omega=kwargs_los['omega_os'],
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
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        # Hessian matrix of the main lens only
        f_xx, f_xy, f_yx, f_yy = super().hessian(x_d, y_d,
                                                 kwargs=kwargs_lens, k=k)

        # Multiply on the left by (1 - Gamma_ds)
        f_xx, f_xy, f_yx, f_yy = self.los.left_multiply(
                                    f_xx, f_xy, f_yx, f_yy,
                                    kappa=kwargs_los['kappa_ds'],
                                    omega=kwargs_los['omega_ds'],
                                    gamma1=kwargs_los['gamma1_ds'],
                                    gamma2=kwargs_los['gamma2_ds'])

        # Multiply on the right by (1 - Gamma_od)
        f_xx, f_xy, f_yx, f_yy = self.los.right_multiply(
                                    f_xx, f_xy, f_yx, f_yy,
                                    kappa=kwargs_los['kappa_od'],
                                    omega=kwargs_los['omega_od'],
                                    gamma1=kwargs_los['gamma1_od'],
                                    gamma2=kwargs_los['gamma2_od'])

        # LOS contribution in the absence of the main lens
        f_xx += kwargs_los['kappa_os'] + kwargs_los['gamma1_os']
        f_xy += kwargs_los['gamma2_os'] - kwargs_los['omega_os']
        f_yx += kwargs_los['gamma2_os'] + kwargs_los['omega_os']
        f_yy += kwargs_los['kappa_os'] - kwargs_los['gamma1_os']
        
        return f_xx, f_xy, f_yx, f_yy
