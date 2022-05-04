__author__ = 'nataliehogg', 'pierrefleury', 'danjohnson98'

from lenstronomy.LensModel.single_plane import SinglePlane
import numpy as np
import copy

__all__ = ['SinglePlaneLOS']


class SinglePlaneLOS(SinglePlane):
    """
    This class is based on the 'SinglePlane' class, modified to include
    line-of-sight effects as presented by Fleury et al. in 2104.08883.
    
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

        kwargs_dominant = [kwarg for i, kwarg in enumerate(kwargs)                                          #DJMod
                       if i != self.index_los]

        return kwargs_dominant, kwargs_los                                                                  #DJMod


    def fermat_potential(self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None):         #DJMod
        """
        Calculates the Fermat Potential with LOS corrections in the tidal regime

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 as a list
        """
        
        kwargs_dominant, kwargs_los = self.split_lens_los(kwargs_lens)

        #the amplification matrices
        A_od = np.array([[1-kwargs_los['kappa_od']-kwargs_los['gamma1_od'],-kwargs_los['gamma2_od']+kwargs_los['omega_od']],[-kwargs_los['gamma2_od']-kwargs_los['omega_od'],1-kwargs_los['kappa_od']+kwargs_los['gamma1_od']]])
        A_os = np.array([[1-kwargs_los['kappa_os']-kwargs_los['gamma1_os'],-kwargs_los['gamma2_os']+kwargs_los['omega_os']],[-kwargs_los['gamma2_os']-kwargs_los['omega_os'],1-kwargs_los['kappa_os']+kwargs_los['gamma1_os']]])
        A_ds = np.array([[1-kwargs_los['kappa_ds']-kwargs_los['gamma1_ds'],-kwargs_los['gamma2_ds']+kwargs_los['omega_ds']],[-kwargs_los['gamma2_ds']-kwargs_los['omega_ds'],1-kwargs_los['kappa_ds']+kwargs_los['gamma1_ds']]])

        #the inverse and transposed amplification matrices
        A_od_tsp = np.transpose(A_od)
        A_ds_inv = np.linalg.inv(A_ds)
        A_os_inv = np.linalg.inv(A_os)

        #the composite amplification matrices
        A_LOS = np.dot(np.dot(A_od_tsp,A_ds_inv),A_os)
        
        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x_image, y_image,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        #Evaluating the potential of the main lens at this position
        effective_potential = super().potential(x_d, y_d, kwargs=kwargs_dominant, k=k)

        #obtaining the source position
        if x_source is None or y_source is None:
            x_source, y_source = self.ray_shooting(x_image, y_image, kwargs_lens, k=k)

        #the source position, modified by A_os_inv
        b_x = A_os_inv[0][0]*x_source + A_os_inv[0][1]*y_source
        b_y = A_os_inv[1][0]*x_source + A_os_inv[1][1]*y_source   

        #alpha'
        f_x = x_image - b_x
        f_y = y_image - b_y       

        #alpha' must then be further distorted by A_LOS
        a_x = A_LOS[0][0]*f_x + A_LOS[0][1]*f_y
        a_y = A_LOS[1][0]*f_x + A_LOS[1][1]*f_y
        
        #we can then obtain the geometrical term
        geometry = (f_x*a_x + f_y*a_y) / 2
        
        #return geometry - effective_potential
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
        
        kwargs_dominant, kwargs_los = self.split_lens_los(kwargs)                          #DJMod

        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        # Displacement due to the main lens only
        f_x, f_y = super().alpha(x_d, y_d, kwargs=kwargs_dominant, k=k)                     #DJMod

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

        kwargs_dominant, kwargs_los = self.split_lens_los(kwargs)                          #DJMod

        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        # Hessian matrix of the main lens only
        f_xx, f_xy, f_yx, f_yy = super().hessian(x_d, y_d,
                                                 kwargs=kwargs_dominant, k=k)              #DJMod

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
