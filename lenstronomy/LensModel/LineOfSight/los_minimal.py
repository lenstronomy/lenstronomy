__author__ = 'nataliehogg', 'pierrefleury'

__all__ = ['LOS_MINIMAL']

class LOSMinimal(object):
    """
    Class allowing one to add tidal line-of-sight effects (convergence and
    shear) to single-plane lensing. Stricly speaking, this is not a profile,
    but when present in list of lens models, it is automatically recognised by
    ModelAPI(), which sets the flag los_effects to True, and thereby leads
    LensModel to use SinglePlaneLOS() instead of SinglePlane(). It is however
    incompatible with MultiPlane().

    This class follows the same structure as LOS, but implements the so-called
    minimal lens model of https://arxiv.org/abs/2104.08883, equation 3.17.

    The key-word arguments are the two components of the two line-of-sight shears,
    all defined with the convention of https://arxiv.org/abs/2104.08883:
    gamma1_od, gamma2_od, gamma1_los, gamma2_los.

    Because LOSMinimal is not a profile, it does not contain the usual functions
    function(), derivatives(), and hessian(), but rather modifies the
    behaviour of those functions in the SinglePlaneLOS() class.

    Instead, it contains the essential building blocks of this modification.
    """

    param_names = ['gamma1_od','gamma2_od',
                   'gamma1_los','gamma2_los']
    lower_limit_default = {pert: -0.5 for pert in param_names}
    upper_limit_default = {pert: 0.5 for pert in param_names}


    def __init__(self, *args, **kwargs):
        self._static = False


    def distort_vector(self, x, y, gamma1=0, gamma2=0):
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
            1 - \\gamma_1 & -\\gamma_2

            -\\gamma_2 & 1 + \\gamma_1
            \\end{pmatrix}
            \\begin{pmatrix}
            x

            y
            \\end{pmatrix}
        """

        x_ = (1 - gamma1) * x - gamma2 * y
        y_ = (1 + gamma1) * y - gamma2 * x

        return x_, y_


    def left_multiply(self, f_xx, f_xy, f_yx, f_yy, gamma1, gamma2):

        """
        Left-multiplies the Hessian matrix of a lens with a distortion matrix
        containing the shears gamma1, gamma2:

        .. math::
            \\mathsf{H}'
            =
            \\begin{pmatrix}
            1 - \\gamma_1 & -\\gamma_2

            -\\gamma_2 & 1 + \\gamma_1
            \\end{pmatrix}
            \\mathsf{H}
        """

        f__xx = (1 - gamma1) * f_xx - gamma2 * f_yx
        f__xy = (1 - gamma1) * f_xy - gamma2 * f_yy
        f__yx = - gamma2 * f_xx + (1 + gamma1) * f_yx
        f__yy = - gamma2 * f_xy + (1 + gamma1) * f_yy

        return f__xx, f__xy, f__yx, f__yy


    def right_multiply(self, f_xx, f_xy, f_yx, f_yy, gamma1, gamma2):

        """
        Right-multiplies the Hessian matrix of a lens with a distortion matrix
        with the shears gamma1, gamma2:

        .. math::
            \\mathsf{H}'
            =
            \\mathsf{H}
            \\begin{pmatrix}
            1 - \\gamma_1 & -\\gamma_2

            -\\gamma_2 & 1 + \\gamma_1
            \\end{pmatrix}
        """

        f__xx = (1 - gamma1) * f_xx - gamma2 * f_xy
        f__xy = - gamma2 * f_xx + (1 + gamma1) * f_xy
        f__yx = (1 - gamma1) * f_yx - gamma2 * f_yy
        f__yy = - gamma2 * f_yx + (1 + gamma1) * f_yy

        return f__xx, f__xy, f__yx, f__yy
