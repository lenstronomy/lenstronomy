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
    - fermat potential
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

        super().__init__(lens_model_list)
        # NB: It is important to run that init first, in order to create a
        # list_func for the entire model, before splitting it between a main
        # lens and the LOS corrections

        # Extract the los model and import its class
        self.index_los = index_los
        self.los_model = lens_model_list[index_los]
        self.los = self._import_class(self.los_model, custom_class=None, kwargs_interp=None)

        # Define a separate class for the main lens
        lens_model_list_wo_los = [
            model for i, model in enumerate(lens_model_list)
            if i != index_los]
        self.main_lens = SinglePlane(lens_model_list_wo_los,
                                     numerical_alpha_class=numerical_alpha_class,
                                     lens_redshift_list=lens_redshift_list,
                                     z_source_convention=z_source_convention,
                                     kwargs_interp=kwargs_interp)

    def split_lens_los(self, kwargs):
        """
        This function splits the list of key-word arguments given to the lens
        model into those that correspond to the lens itself (kwargs_main), and
        those that correspond to the line-of-sight corrections (kwargs_los).

        :param kwargs: the list of key-word arguments passed to lenstronomy
        :return: a list of kwargs corresponding to the lens and a list of kwargs corresponding to the LOS effects
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

        kwargs_main = [kwarg for i, kwarg in enumerate(kwargs)
                       if i != self.index_los]

        return kwargs_main, kwargs_los


    def fermat_potential(self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None):
        """
        Calculates the Fermat Potential with LOS corrections in the tidal regime

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 as a list
        """

        kwargs_main, kwargs_los = self.split_lens_los(kwargs_lens)

        # the amplification matrices
        A_od = np.array([[1-kwargs_los['kappa_od']-kwargs_los['gamma1_od'],-kwargs_los['gamma2_od']+kwargs_los['omega_od']],[-kwargs_los['gamma2_od']-kwargs_los['omega_od'],1-kwargs_los['kappa_od']+kwargs_los['gamma1_od']]])
        A_os = np.array([[1-kwargs_los['kappa_os']-kwargs_los['gamma1_os'],-kwargs_los['gamma2_os']+kwargs_los['omega_os']],[-kwargs_los['gamma2_os']-kwargs_los['omega_os'],1-kwargs_los['kappa_os']+kwargs_los['gamma1_os']]])
        A_ds = np.array([[1-kwargs_los['kappa_ds']-kwargs_los['gamma1_ds'],-kwargs_los['gamma2_ds']+kwargs_los['omega_ds']],[-kwargs_los['gamma2_ds']-kwargs_los['omega_ds'],1-kwargs_los['kappa_ds']+kwargs_los['gamma1_ds']]])

        # the inverse and transposed amplification matrices
        A_od_tsp = np.transpose(A_od)
        A_ds_inv = np.linalg.inv(A_ds)
        A_os_inv = np.linalg.inv(A_os)

        # the composite amplification matrices
        A_LOS = np.dot(np.dot(A_od_tsp,A_ds_inv),A_os)

        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x_image, y_image,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        # Evaluating the potential of the main lens at this position
        effective_potential = self.main_lens.potential(x_d, y_d, kwargs=kwargs_main, k=k)

        # obtaining the source position
        if x_source is None or y_source is None:
            x_source, y_source = self.ray_shooting(x_image, y_image, kwargs_lens, k=k)

        # the source position, modified by A_os_inv
        b_x = A_os_inv[0][0]*x_source + A_os_inv[0][1]*y_source
        b_y = A_os_inv[1][0]*x_source + A_os_inv[1][1]*y_source

        # alpha'
        f_x = x_image - b_x
        f_y = y_image - b_y

        # alpha' must then be further distorted by A_LOS
        a_x = A_LOS[0][0]*f_x + A_LOS[0][1]*f_y
        a_y = A_LOS[1][0]*f_x + A_LOS[1][1]*f_y

        # we can then obtain the geometrical term
        geometry = (f_x*a_x + f_y*a_y) / 2

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

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)

        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        # Displacement due to the main lens only
        f_x, f_y = self.main_lens.alpha(x_d, y_d, kwargs=kwargs_main, k=k)

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

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)

        # Angular position where the ray hits the deflector's plane
        x_d, y_d = self.los.distort_vector(x, y,
                                           kappa=kwargs_los['kappa_od'],
                                           omega=kwargs_los['omega_od'],
                                           gamma1=kwargs_los['gamma1_od'],
                                           gamma2=kwargs_los['gamma2_od'])

        # Hessian matrix of the main lens only
        f_xx, f_xy, f_yx, f_yy = self.main_lens.hessian(x_d, y_d,
                                                 kwargs=kwargs_main, k=k)

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


    def mass_3d(self, r, kwargs, bool_list=None):
        """
        Computes the mass within a 3d sphere of radius r *for the main lens only*

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass (in angular units, modulo epsilon_crit)
        """

        print("Note: The computation of the 3d mass ignores the LOS corrections.")

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        mass_3d = self.main_lens.mass_3d(r=r, kwargs=kwargs_main, bool_list=bool_list)

        return mass_3d


    def mass_2d(self, r, kwargs, bool_list=None):
        """
        Computes the mass enclosed a projected (2d) radius r *for the main lens only*

        The mass definition is such that:

        .. math::
            \\alpha = mass_2d / r / \\pi

        with alpha is the deflection angle

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: projected mass (in angular units, modulo epsilon_crit)
        """

        print("Note: The computation of the 2d mass ignores the LOS corrections.")

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        mass_2d = self.main_lens.mass_2d(r=r, kwargs=kwargs_main, bool_list=bool_list)

        return mass_2d


    def density(self, r, kwargs, bool_list=None):
        """
        3d mass density at radius r *for the main lens only*
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass density at radius r (in angular units, modulo epsilon_crit)
        """

        print("Note: The computation of the density ignores the LOS corrections.")

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        density = self.main_lens.density(r=r, kwargs=kwargs_main, bool_list=bool_list)

        return density


    def potential(self, x, y, kwargs, k=None):
        """
        Lensing potential *of the main lens only*
        In the presence of LOS corrections, the system generally does not admit
        a potential, in the sense that the curl of alpha is generally non-zero

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param k: only evaluate the k-th lens model
        :return: lensing potential in units of arcsec^2
        """

        print("Note: The computation of the potential ignores the LOS corrections.\
              In the presence of LOS corrections, a lensing system does not always\
              derive from a potential.")

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        potential = self.main_lens.potential(x, y, kwargs, k=k)

        return potential
