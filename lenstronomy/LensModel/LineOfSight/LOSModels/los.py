__author__ = 'pierrefleury'

#import numpy as np

__all__ = ['LOS']

class LOS(object):
    # PF: update the docstring at the end of implementation.
    """
    Class allowing one to add tidal line-of-sight effects (convergence and
    shear) to single-plane lensing. Stricly speaking, this is not a profile,
    but when present in list of lens models, it is automatically recognised by
    ModelAPI(), which sets the flag los_effects to True, and thereby leads
    LensModel to use SinglePlaneLOS() instead of SinglePlane(). It is however
    incompatible with MultiPlane().
    
    The key-word arguments are the three line-of-sight convergences, the
    two components of the three line-of-sight shears, and the three
    line-of-sight rotations, all defined with the convention of
    https://arxiv.org/abs/2104.08883:
    kappa_od, kappa_os, kappa_ds, gamma1_od, gamma2_od, gamma1_os, gamma2_os,
    gamma1_ds, gamma2_ds, omega_od, omega_os, omega_ds
    
    Because LOS is not a profile, it does not contain the usual functions
    function(), derivatives(), and hessian(), but rather modifies the
    behaviour of those functions in the SinglePlaneLOS() class.
    
    Instead, it contains the essential building blocks of this modification.
    """
    
    param_names = ['kappa_od', 'kappa_os', 'kappa_ds',
                   'gamma1_od','gamma2_od',
                   'gamma1_os','gamma2_os',
                   'gamma1_ds','gamma2_ds',
                   'omega_od', 'omega_os', 'omega_ds']
    lower_limit_default = {pert: -0.5 for pert in param_names}
    upper_limit_default = {pert: 0.5 for pert in param_names}
    
    
    def __init__(self, *args, **kwargs):
        self._static = False
    
    
    def distort_vector(self, x, y, kappa=0, gamma1=0, gamma2=0, omega=0):
        """
        This function applies a distortion matrix to a vector (x, y) and 
        returns (x', y') as follows:
            
        .. math::
            \\begin{pmatrix}
            x'
            
            y'
            \\end{pmatrix}
            =
            \\begin{pmatrix}
            1 - \\kappa - \\gamma_1 & -\\gamma_2 + \\omega

            -\\gamma_2 - \\omega & 1 - \\kappa + \\gamma_1
            \\end{pmatrix}
            \\begin{pmatrix}
            x
            
            y
            \\end{pmatrix}
        """
        
        x_ = (1 - kappa - gamma1) * x + (-gamma2 + omega) * y
        y_ = (1 - kappa + gamma1) * y - (gamma2 + omega) * x
        
        return x_, y_
    
    
    def left_multiply(self, f_xx, f_xy, f_yx, f_yy, 
                      kappa, gamma1, gamma2, omega):
        
        """
        Left-multiplies the Hessian matrix of a lens with a distortion matrix
        with convergence kappa, shear gamma1, gamma2, and rotation omega:
            
        .. math::
            \\mathsf{H}'
            =
            \\begin{pmatrix}
            1 - \\kappa - \\gamma_1 & -\\gamma_2 + \\omega

            -\\gamma_2 - \\omega & 1 - \\kappa + \\gamma_1
            \\end{pmatrix}
            \\mathsf{H}
        """
        
        f__xx = (1 - kappa - gamma1) * f_xx + (- gamma2 + omega) * f_yx
        f__xy = (1 - kappa - gamma1) * f_xy + (- gamma2 + omega) * f_yy
        f__yx = - (gamma2 + omega) * f_xx + (1 - kappa + gamma1) * f_yx
        f__yy = - (gamma2 + omega) * f_xy + (1 - kappa + gamma1) * f_yy
        
        return f__xx, f__xy, f__yx, f__yy
    
    
    def right_multiply(self, f_xx, f_xy, f_yx, f_yy, 
                      kappa, gamma1, gamma2, omega):
        
        """
        Right-multiplies the Hessian matrix of a lens with a distortion matrix
        with convergence kappa and shear gamma1, gamma2:
            
        .. math::
            \\mathsf{H}'
            =
            \\mathsf{H}
            \\begin{pmatrix}
            1 - \\kappa - \\gamma_1 & -\\gamma_2 + \\omega

            -\\gamma_2 - \\omega & 1 - \\kappa + \\gamma_1
            \\end{pmatrix}
        """
        
        f__xx = (1 - kappa - gamma1) * f_xx - (gamma2 + omega) * f_xy
        f__xy = (- gamma2 + omega) * f_xx + (1 - kappa + gamma1) * f_xy
        f__yx = (1 - kappa - gamma1) * f_yx - (gamma2 + omega) * f_yy
        f__yy = (- gamma2 + omega) * f_yx + (1 - kappa + gamma1) * f_yy
        
        return f__xx, f__xy, f__yx, f__yy
        