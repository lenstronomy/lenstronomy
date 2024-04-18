__author__ = ['TheoDuboscq']

from lenstronomy.LensModel.single_plane import SinglePlane
import numpy as np
import copy

__all__ = ['SinglePlaneLOSFlexion']


class SinglePlaneLOSFlexion(SinglePlane):
    """
    This class is based on the 'SinglePlane' class, modified to include
    line-of-sight flexion effects.

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

    def __init__(self, lens_model_list, index_losf,                             
                 numerical_alpha_class=None,
                 lens_redshift_list=None,
                 z_source_convention=None,
                 kwargs_interp=None):
        """
        Instance of SinglePlaneLOSFlexion() based on the SinglePlane(), except:
        - argument "index_losf" indicating the position of the LOSF model in the
        lens_model_list (for correct association with kwargs)
        - attribute "losf" containing the LOSF model.
        """

        super(SinglePlaneLOSFlexion, self).__init__(lens_model_list)
        # NB: It is important to run that init first, in order to create a
        # list_func for the entire model, before splitting it between a main
        # lens and the LOS flexion corrections

        # Extract the los flexion model and import its class
        self._index_losf = index_losf
        self._losf_model = lens_model_list[index_losf]
        self.losf = self._import_class(self._losf_model, custom_class=None, kwargs_interp=None)

        # Define a separate class for the main lens
        lens_model_list_wo_los = [
            model for i, model in enumerate(lens_model_list)
            if i != index_losf]
        self._main_lens = SinglePlane(lens_model_list_wo_los,
                                      numerical_alpha_class=numerical_alpha_class,
                                      lens_redshift_list=lens_redshift_list,
                                      z_source_convention=z_source_convention,
                                      kwargs_interp=kwargs_interp)
                                      

    def split_lens_losf(self, kwargs): 
        """
        This function splits the list of key-word arguments given to the lens
        model into those that correspond to the lens itself (kwargs_main), and
        those that correspond to the line-of-sight corrections (kwargs_losf),
        including line-of-sight flexion.

        :param kwargs: the list of key-word arguments passed to lenstronomy
        :return: a list of kwargs corresponding to the lens and a list of kwargs corresponding to the LOS effects
        """

        kwargs_losf = copy.deepcopy(kwargs[self._index_losf])
        # if 'LOSF_MINIMAL' is at play, we set kappa_os = kappa_los, gamma_os = gamma_los, F_os = F_los, G_os = G_los,
        # F_1ds = F_1los, G_1ds = G_1los, kappa_ds = kappa_od, gamma_ds = gamma_od, F_2ds = F_od, G_2ds = G_od.
        # In the following we convert a list expressed for the LOSF_MINIMAL model as a list expressed for the LOSF model,
        # that's why for instance we delete the kappa_los entry while creating the kappa_os entry
        if self._losf_model == 'LOSF_MINIMAL':
            kwargs_losf['kappa_os'] = kwargs_losf.pop('kappa_los')      
            kwargs_losf['gamma1_os'] = kwargs_losf.pop('gamma1_los')
            kwargs_losf['gamma2_os'] = kwargs_losf.pop('gamma2_los')
            kwargs_losf['F1_os'] = kwargs_losf.pop('F1_los')
            kwargs_losf['F2_os'] = kwargs_losf.pop('F2_los')
            kwargs_losf['G1_os'] = kwargs_losf.pop('G1_los')
            kwargs_losf['G2_os'] = kwargs_losf.pop('G2_los')
            kwargs_losf['F1_1ds'] = kwargs_losf.pop('F1_1los')
            kwargs_losf['F2_1ds'] = kwargs_losf.pop('F2_1los')
            kwargs_losf['G1_1ds'] = kwargs_losf.pop('G1_1los')
            kwargs_losf['G2_1ds'] = kwargs_losf.pop('G2_1los')
            kwargs_losf['kappa_ds'] = kwargs_losf['kappa_od']    # here kappa_od is still present in the list of arguments for LOSF, thus we don't delete
            kwargs_losf['gamma1_ds'] = kwargs_losf['gamma1_od']
            kwargs_losf['gamma2_ds'] = kwargs_losf['gamma2_od']
            kwargs_losf['F1_2ds'] = kwargs_losf['F1_od']
            kwargs_losf['F2_2ds'] = kwargs_losf['F2_od']
            kwargs_losf['G1_2ds'] = kwargs_losf['G1_od']
            kwargs_losf['G2_2ds'] = kwargs_losf['G2_od']
            kwargs_losf['omega_os'] = kwargs_losf['omega_los']

        kwargs_main = [kwarg for i, kwarg in enumerate(kwargs)
                       if i != self._index_losf]

        return kwargs_main, kwargs_losf
        

    def fermat_potential(self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None): 
        """
        Calculates the Fermat Potential with LOS corrections in the flexion regime

        :param x_image: image position
        :param y_image: image position
        :param x_source: source position
        :param y_source: source position
        :param kwargs_lens: list of keyword arguments of lens model parameters matching the lens model classes
        :return: fermat potential in arcsec**2 as a list
        """
        # Beware! Here the formula used for the computation of the fermat potential is different than the one presented in the paper "Weak lensing of strong lensing: beyond the tidal regime",
        # but it is equivalent. This form has the advantage of being more compact. 

        kwargs_main, kwargs_losf = self.split_lens_losf(kwargs_lens)
        theta = x_image + 1j*y_image
        thetac = theta.conjugate()
        
        
        # F_los, G_los, F_1los, G_1los, kappa_los, gamma_los
        F_los = kwargs_losf['F1_od'] + kwargs_losf['F1_os'] - kwargs_losf['F1_2ds'] + 1j*(kwargs_losf['F2_od'] + kwargs_losf['F2_os'] - kwargs_losf['F2_2ds'])
        G_los = kwargs_losf['G1_od'] + kwargs_losf['G1_os'] - kwargs_losf['G1_2ds'] + 1j*(kwargs_losf['G2_od'] + kwargs_losf['G2_os'] - kwargs_losf['G2_2ds'])
        
        F_1los = kwargs_losf['F1_od'] + kwargs_losf['F1_1ds'] - kwargs_losf['F1_2ds'] + 1j*(kwargs_losf['F2_od'] + kwargs_losf['F2_1ds'] - kwargs_losf['F2_2ds'])
        G_1los = kwargs_losf['G1_od'] + kwargs_losf['G1_1ds'] - kwargs_losf['G1_2ds'] + 1j*(kwargs_losf['G2_od'] + kwargs_losf['G2_1ds'] - kwargs_losf['G2_2ds'])
        
        kappa_los = kwargs_losf['kappa_od'] + kwargs_losf['kappa_os'] - kwargs_losf['kappa_ds']
        gamma_los = kwargs_losf['gamma1_od'] + kwargs_losf['gamma1_os'] - kwargs_losf['gamma1_ds'] + 1j*(kwargs_losf['gamma2_od'] + kwargs_losf['gamma2_os'] - kwargs_losf['gamma2_ds'])
        
        
        # computation of f(theta) 
        theta_d = (1 - kwargs_losf['kappa_od'])*theta - (kwargs_losf['gamma1_od'] +1j*kwargs_losf['gamma2_od'])*thetac - 1/2*( (kwargs_losf['F1_od'] - 1j*kwargs_losf['F2_od'])*theta**2 \
                   + 2*(kwargs_losf['F1_od'] + 1j*kwargs_losf['F2_od'])*theta*thetac + (kwargs_losf['G1_od'] + 1j*kwargs_losf['G2_od'])*thetac**2 )
        x_d, y_d = theta_d.real, theta_d.imag
        
        
        # Evaluating the potential of the main lens at this position
        supF = F_los*theta*thetac**2
        supG = G_los*thetac**3
        effective_potential = self._main_lens.potential(x_d, y_d, kwargs=kwargs_main, k=k) + 1/2*supF.real + 1/6*supG.real
        
        
        # alpha_ods(f(theta))
        m_x, m_y = self._main_lens.alpha(x_d, y_d, kwargs=kwargs_main, k=k) 
        
        
        # alpha_eff, different from the one of the paper. In this one we add some flexion in alpha_eff (group3 term) which makes the global expression more compact.
        alpha1_m, alpha2_m = self._main_lens.alpha(x_image, y_image, kwargs=kwargs_main, k=k) 
        alpha_m = alpha1_m + 1j*alpha2_m
        alpha_mc = alpha_m.conjugate()
        
        group1 = kwargs_losf['kappa_od'] + (kwargs_losf['F1_od'] - 1j*kwargs_losf['F2_od'])*theta + (kwargs_losf['F1_od'] + 1j*kwargs_losf['F2_od'])*thetac
        group2 = kwargs_losf['gamma1_od'] + 1j*kwargs_losf['gamma2_od'] + (kwargs_losf['F1_od'] + 1j*kwargs_losf['F2_od'])*theta + (kwargs_losf['G1_od'] + 1j*kwargs_losf['G2_od'])*thetac
        group3 = F_los.conjugate()*theta**2 + 2*F_los*theta*thetac + G_los*thetac**2
        alpha_eff = m_x + 1j*m_y - group1*alpha_m - group2*alpha_mc + 1/2*group3
        alpha_effc = alpha_eff.conjugate()
        
        
        # geometric term
        geomc = alpha_effc * ( (1 + kappa_los)*alpha_eff + gamma_los*alpha_effc + 2/3*( F_1los.conjugate()*alpha_eff**2 + 2*F_1los*alpha_eff*alpha_effc + G_1los*alpha_effc**2 ) )
        geometry = geomc.real/2
        

        return geometry - effective_potential



    def alpha(self, x, y, kwargs, k=None):   
        """
        Displacement angle including the line-of-sight flexion corrections

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters
         matching the lens model classes, including line-of-sight flexion corrections
        :param k: only evaluate the k-th lens model
        :return: deflection angles in units of arcsec
        """

        kwargs_main, kwargs_losf = self.split_lens_losf(kwargs)
        theta = x + y*1j
        thetac = theta.conjugate()
        

        # Angular position where the ray hits the deflector's plane
        theta_d = (1 - kwargs_losf['kappa_od'])*theta - (kwargs_losf['gamma1_od'] +1j*kwargs_losf['gamma2_od'])*thetac - 1/2*( (kwargs_losf['F1_od'] - 1j*kwargs_losf['F2_od'])*theta**2 \
                   + 2*(kwargs_losf['F1_od'] + 1j*kwargs_losf['F2_od'])*abs(theta)**2 + (kwargs_losf['G1_od'] + 1j*kwargs_losf['G2_od'])*thetac**2 )
        x_d, y_d = theta_d.real, theta_d.imag

        # Displacement due to the main lens only
        m_x, m_y = self._main_lens.alpha(x_d, y_d, kwargs=kwargs_main, k=k)  

        # Flexed alpha_os
        alpha_os = kwargs_losf['kappa_os']*theta + 1j*kwargs_losf['omega_os']*theta + (kwargs_losf['gamma1_os'] +1j*kwargs_losf['gamma2_os'])*thetac + 1/2*( (kwargs_losf['F1_os'] - 1j*kwargs_losf['F2_os'])*theta**2 \
                   + 2*(kwargs_losf['F1_os'] + 1j*kwargs_losf['F2_os'])*abs(theta)**2 + (kwargs_losf['G1_os'] + 1j*kwargs_losf['G2_os'])*thetac**2 )
        x_s, y_s = alpha_os.real, alpha_os.imag

        # Terms in alpha_ods(theta)
        alpha_ods_x, alpha_ods_y = self._main_lens.alpha(x, y, kwargs=kwargs_main, k=k)
        alpha_ods = alpha_ods_x + 1j*alpha_ods_y
        lin = - (kwargs_losf['kappa_ds'] + (kwargs_losf['F1_2ds'] - 1j*kwargs_losf['F2_2ds'])*theta + (kwargs_losf['F1_2ds'] + 1j*kwargs_losf['F2_2ds'])*thetac) * alpha_ods \
              - (kwargs_losf['gamma1_ds'] + 1j*kwargs_losf['gamma2_ds'] + (kwargs_losf['F1_2ds'] + 1j*kwargs_losf['F2_2ds'])*theta + (kwargs_losf['G1_2ds'] + 1j*kwargs_losf['G2_2ds'])*thetac) \
               * alpha_ods.conjugate()
        lin_x, lin_y = lin.real, lin.imag
        
        # Quadratic terms in alpha_ods
        quad = 1/2*( (kwargs_losf['F1_1ds'] - 1j*kwargs_losf['F2_1ds'])*alpha_ods**2 + 2*(kwargs_losf['F1_1ds'] + 1j*kwargs_losf['F2_1ds'])*alpha_ods*alpha_ods.conjugate() \
               + (kwargs_losf['G1_1ds'] + 1j*kwargs_losf['G2_1ds'])*alpha_ods.conjugate()**2 )
        quad_x, quad_y = quad.real, quad.imag

        # Complete displacement
        m_x += x_s + lin_x + quad_x
        m_y += y_s + lin_y + quad_y

        return m_x, m_y
        
        

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

        kwargs_main, kwargs_losf = self.split_lens_losf(kwargs)
        theta = x + y*1j
        thetac = theta.conjugate()
        
        alpha_ods_x, alpha_ods_y = self._main_lens.alpha(x, y, kwargs=kwargs_main, k=k)
        alpha_ods = alpha_ods_x + 1j*alpha_ods_y
        
        # Computation of complex parameters
        F_od = kwargs_losf['F1_od'] + 1j*kwargs_losf['F2_od']
        G_od = kwargs_losf['G1_od'] + 1j*kwargs_losf['G2_od']
        
        F_1ds = kwargs_losf['F1_1ds'] + 1j*kwargs_losf['F2_1ds']
        F_2ds = kwargs_losf['F1_2ds'] + 1j*kwargs_losf['F2_2ds']
        
        G_1ds = kwargs_losf['G1_1ds'] + 1j*kwargs_losf['G2_1ds']
        G_2ds = kwargs_losf['G1_2ds'] + 1j*kwargs_losf['G2_2ds']
        
        F_os = kwargs_losf['F1_os'] + 1j*kwargs_losf['F2_os']
        G_os = kwargs_losf['G1_os'] + 1j*kwargs_losf['G2_os']
        

        # Angular position where the ray hits the deflector's plane
        theta_d = (1 - kwargs_losf['kappa_od'])*theta - (kwargs_losf['gamma1_od'] +1j*kwargs_losf['gamma2_od'])*thetac - 1/2*( (kwargs_losf['F1_od'] - 1j*kwargs_losf['F2_od'])*theta**2 \
                   + 2*(kwargs_losf['F1_od'] + 1j*kwargs_losf['F2_od'])*abs(theta)**2 + (kwargs_losf['G1_od'] + 1j*kwargs_losf['G2_od'])*thetac**2 )
        x_d, y_d = theta_d.real, theta_d.imag
        
        # Hessian matrix of the main lens only
        f_xx, f_xy, f_yx, f_yy = self._main_lens.hessian(x_d, y_d, kwargs=kwargs_main, k=k)
        
        
        # Computation of Gamma_ds
        kappa_ds = kwargs_losf['kappa_ds'] + F_2ds.conjugate()*theta + F_2ds*thetac - F_1ds.conjugate()*alpha_ods - F_1ds*alpha_ods.conjugate()
        gamma_ds = kwargs_losf['gamma1_ds'] + 1j*kwargs_losf['gamma2_ds'] + F_2ds*theta + G_2ds*thetac - F_1ds*alpha_ods - G_1ds*alpha_ods.conjugate()
        gamma1_ds, gamma2_ds = gamma_ds.real, gamma_ds.imag

        # Multiply on the left by (1 - Gamma_ds)
        f__xx = (1 - kappa_ds - gamma1_ds) * f_xx - gamma2_ds * f_yx
        f__xy = (1 - kappa_ds - gamma1_ds) * f_xy - gamma2_ds * f_yy
        f__yx = - gamma2_ds * f_xx + (1 - kappa_ds + gamma1_ds) * f_yx
        f__yy = - gamma2_ds * f_xy + (1 - kappa_ds + gamma1_ds) * f_yy
        
                                    
	# Computation of Gamma_od
        kappa_od = kwargs_losf['kappa_od'] + F_od.conjugate()*theta + F_od*thetac
        gamma_od = kwargs_losf['gamma1_od'] + 1j*kwargs_losf['gamma2_od'] + F_od*theta + G_od*thetac
        gamma1_od, gamma2_od = gamma_od.real, gamma_od.imag

        # Multiply on the right by (1 - Gamma_od)
        f_xx = (1 - kappa_od - gamma1_od) * f__xx - gamma2_od * f__xy
        f_xy = - gamma2_od * f__xx + (1 - kappa_od + gamma1_od) * f__xy
        f_yx = (1 - kappa_od - gamma1_od) * f__yx - gamma2_od * f__yy
        f_yy = - gamma2_od * f__yx + (1 - kappa_od + gamma1_od) * f__yy
        
                                    
        # Computation of Gamma_os
        kappa_os = kwargs_losf['kappa_os'] + F_os.conjugate()*theta + F_os*thetac - F_2ds.conjugate()*alpha_ods - F_2ds*alpha_ods.conjugate()
        gamma_os = kwargs_losf['gamma1_os'] +1j*kwargs_losf['gamma2_os'] + F_os*theta + G_os*thetac - F_2ds*alpha_ods - G_2ds*alpha_ods.conjugate()
        gamma1_os, gamma2_os = gamma_os.real, gamma_os.imag

        # LOS contribution in the absence of the main lens
        f_xx += kappa_os + gamma1_os
        f_xy += gamma2_os
        f_yx += gamma2_os
        f_yy += kappa_os - gamma1_os

        return f_xx.real, f_xy.real, f_yx.real, f_yy.real
        

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
        mass_3d = self._main_lens.mass_3d(r=r, kwargs=kwargs_main, bool_list=bool_list)

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
        mass_2d = self._main_lens.mass_2d(r=r, kwargs=kwargs_main, bool_list=bool_list)

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
        density = self._main_lens.density(r=r, kwargs=kwargs_main, bool_list=bool_list)

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

        # kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        potential = self._main_lens.potential(x, y, kwargs, k=k)

        return potential
