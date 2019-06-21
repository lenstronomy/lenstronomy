from lenstronomy.LensModel.Profiles.flexion import Flexion

class Flexionfg(object):
    """
    Flexion consist of basis F flexion and G flexion (F1,F2,G1,G2),
    see formulas 2.54, 2.55 in Massimo Meneghetti 2017 - "Introduction to Gravitational Lensing".
    """
    param_names = ['F1', 'F2', 'G1', 'G2', 'ra_0', 'dec_0']
    lower_limit_default = {'F1': -0.1, 'F2': -0.1, 'G1': -0.1, 'G2': -0.1, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'F1': 0.1, 'F2': 0.1, 'G1': 0.1, 'G2': 0.1, 'ra_0': 100, 'dec_0': 100}

    def __init__(self):
        self.flexion_cart = Flexion()

    def function(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        """
        lensing potential

        :param x: x-coordinate
        :param y: y-coordinate
        :param F1: F1 flexion, derivative of kappa in x direction
        :param F2: F2 flexion, derivative of kappa in y direction
        :param G1: G1 flexion
        :param G2: G2 flexion
        :param ra_0: center x-coordinate
        :param dec_0: center y-coordinate
        :return: lensing potential
        """
        _g1, _g2, _g3, _g4 = self.transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.function(x,y,_g1, _g2, _g3, _g4, ra_0, dec_0)

    def derivatives(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        """
        deflection angle
        :param x: x-coordinate
        :param y: y-coordinate
        :param F1: F1 flexion, derivative of kappa in x direction
        :param F2: F2 flexion, derivative of kappa in y direction
        :param G1: G1 flexion
        :param G2: G2 flexion
        :param ra_0: center x-coordinate
        :param dec_0: center x-coordinate
        :return: deflection angle
        """
        _g1, _g2, _g3, _g4 = self.transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.derivatives(x, y, _g1, _g2, _g3, _g4, ra_0, dec_0)

    def hessian(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        """
        Hessian matrix
        :param x: x-coordinate
        :param y: y-coordinate
        :param F1: F1 flexion, derivative of kappa in x direction
        :param F2: F2 flexion, derivative of kappa in y direction
        :param G1: G1 flexion
        :param G2: G2 flexion
        :param ra_0: center x-coordinate
        :param dec_0: center y-coordinate
        :return: second order derivatives f_xx, f_yy, f_xy
        """
        _g1, _g2, _g3, _g4 = self.transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.hessian(x, y, _g1, _g2, _g3, _g4, ra_0, dec_0)

    def transform_fg(self, F1, F2, G1, G2):
        """
        basis transform from (F1,F2,G1,G2) to (g1,g2,g3,g4)
        :param f1: F1 flexion, derivative of kappa in x direction
        :param f2: F2 flexion, derivative of kappa in y direction
        :param g1: G1 flexion
        :param g2: G2 flexion
        :return: g1,g2,g3,g4 (phi_xxx, phi_xxy, phi_xyy, phi_yyy)
        """
        g1 = (3*F1 + G1) * 0.5
        g2 = (3*F2 + G2) * 0.5
        g3 = (F1 - G1) * 0.5
        g4 = (F2 - G2) * 0.5
        return g1,g2,g3,g4