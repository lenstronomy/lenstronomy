__author__ = 'pierrefleury'

from lenstronomy.LensModel.Profiles.epl import EPL

__all__ = ['EPL_SHEARED']

class EPL_SHEARED(EPL):
    """
    Sheared version of the elliptic power-law profile.
    If :math:`\\psi_{\\rm EPL}`,
    :math:`\\boldsymbol{\\alpha}_{\\rm EPL}`,
    and
    :math:`\\mathsf{H}_{\\rm EPL}` are respectively the EPL Fermat potential,
    deflection angle, and Hessian matrix, then the sheared EPL is such that
    
    .. math::
        
        \\psi(\\boldsymbol{x})
        &= \\psi_{\\rm EPL}
            \\left[
                \\boldsymbol{x}_{\\rm c} +
                (1-\\mathsf{\\Gamma})
                (\\boldsymbol{x} - \\boldsymbol{x}_{\\rm c})
            \\right]
            
        \\boldsymbol{\\alpha}(\\boldsymbol{x})
        &= (1-\\mathsf{\\Gamma})
            \\boldsymbol{\\alpha}_{\\rm EPL}
           \\left[
               \\boldsymbol{x}_{\\rm c} +
               (1-\\mathsf{\\Gamma})
               (\\boldsymbol{x} - \\boldsymbol{x}_{\\rm c})
           \\right]
         
        \\mathsf{H}(\\boldsymbol{x})
        &= (1-\\mathsf{\\Gamma})
            \\mathsf{H}_{\\rm EPL}
           \\left[
               \\boldsymbol{x}_{\\rm c} +
               (1-\\mathsf{\\Gamma})
               (\\boldsymbol{x} - \\boldsymbol{x}_{\\rm c})
           \\right]
            (1-\\mathsf{\\Gamma})
            
        
    where :math:`\\boldsymbol{x}_{\\rm c}` denotes the centre of the EPL, and
    with the shear matrix
    
    .. math::
        \\mathsf{\\Gamma}
        =
        \\begin{pmatrix}
        \\gamma_1 & \\gamma_2

        \\gamma_2 & -\\gamma_1
        \\end{pmatrix}
        
    which thus adds two parameters :math:`(\\gamma_1, \\gamma_2)` to the
    model.
    
    The fact that the shear matrix is only applied to relative position
    :math:`\\boldsymbol{x} - \\boldsymbol{x}_{\\rm c}` ensures that there is
    no spurious displacement of the image position within the main lens plane.
    """
        
    param_names = ['theta_E', 'gamma',
                   'e1', 'e2', 'gamma1', 'gamma2',
                   'center_x', 'center_y'
                   ]
    lower_limit_default = {'theta_E': 0, 'gamma': 1.5,
                           'e1': -0.5, 'e2': -0.5,
                           'gamma1': -0.5, 'gamma2': -0.5,
                           'center_x': -100, 'center_y': -100
                           }
    upper_limit_default = {'theta_E': 100, 'gamma': 2.5,
                           'e1': 0.5, 'e2': 0.5,
                           'gamma1': 0.5, 'gamma2': 0.5,
                           'center_x': 100, 'center_y': 100}
    
    def shear(self, x, y, gamma1, gamma2, center_x, center_y):
        """
        
        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param gamma1: shear component
        :param gamma2: shear component
        :param center_x: profile center
        :param center_y: profile center
        :returns: position that is sheared with respect to the centre
        """
        # relative position to the centre
        x_ = x - center_x
        y_ = y - center_y
        # shear it
        delta_x = (1 - gamma1) * x_ - gamma2 * y_
        delta_y = (1 + gamma1) * y_ - gamma2 * x_
        # shift it back
        x_new = center_x + delta_x
        y_new = center_x + delta_y
        
        return x_new, y_new
    
    def function(self, x, y,
                 theta_E, gamma, e1, e2,
                 gamma1=0, gamma2=0,
                 center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param gamma1: shear component
        :param gamma2: shear component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        # get sheared position
        x_, y_ = self.shear(x, y, gamma1, gamma2, center_x, center_y)
        # evaluate the potential at this position
        psi = super().function(x_, y_, theta_E, gamma, e1, e2,
                               center_x, center_y)
        
        return psi
    
    def derivatives(self, x, y,
                    theta_E, gamma, e1, e2,
                    gamma1=0, gamma2=0,
                    center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param gamma1: shear component
        :param gamma2: shear component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        # get sheared position
        x_, y_ = self.shear(x, y, gamma1, gamma2, center_x, center_y)
        # evaluate deflection angle
        f_x, f_y = super().derivatives(x_, y_, theta_E, gamma, e1, e2,
                                       center_x, center_y)
        
        # shear the deflection angle
        alpha_x, alpha_y = self.shear(f_x, f_y, gamma1, gamma2,
                                      center_x=0, center_y=0)
        
        return alpha_x, alpha_y

    def hessian(self, x, y,
                theta_E, gamma, e1, e2,
                gamma1=0, gamma2=0,
                center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
        
        # get sheared position
        x_, y_ = self.shear(x, y, gamma1, gamma2, center_x, center_y)
        # evaluate hessian at that position
        f_xx, f_xy, f_yx, f_yy = super().hessian(x_, y_,
                                                 theta_E, gamma, e1, e2,
                                                 center_x, center_y)
        # transform the matrix by applying the shear to it
        f__xx = ( (1 - gamma1)**2 * f_xx
                 - 2 * (1 - gamma1) * gamma2 * f_xy
                 + gamma2**2 * f_yy)
        f__xy = (- (1 - gamma1) * gamma2 * f_xx
                 + (1 + gamma2**2 - gamma1**2) * f_xy
                 - (1 + gamma1) * gamma2 * f_yy)
        f__yy = (  gamma2**2 * f_xx
                 - 2 * (1 + gamma1) * gamma2 * f_xy
                 + (1 + gamma1)**2 * f_yy)

        return f__xx, f__xy, f__xy, f__yy