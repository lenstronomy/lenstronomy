from lenstronomy.LensModel.Profiles.flexion import Flexion

class FlexionFG(object):
    """
    Flexion with F1,F2,G1,G2
    """
    param_names = ['F1', 'F2', 'G1', 'G2', 'ra_0', 'dec_0']
    lower_limit_default = {'F1': -0.1, 'F2': -0.1, 'G1': -0.1, 'G2': -0.1, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'F1': 0.1, 'F2': 0.1, 'G1': 0.1, 'G2': 0.1, 'ra_0': 100, 'dec_0': 100}

    def __init__(self):
        self.flexion_cart = Flexion()

    def function(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        _g1, _g2, _g3, _g4 = self._transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.function(x,y,_g1, _g2, _g3, _g4, ra_0, dec_0)

    def derivatives(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        _g1, _g2, _g3, _g4 = self._transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.derivatives(x, y, _g1, _g2, _g3, _g4, ra_0, dec_0)

    def hessian(self, x, y, F1, F2, G1, G2, ra_0=0, dec_0=0):
        """
        Menegetti equations 2.61 2.62 definitions of flexion
        """
        _g1, _g2, _g3, _g4 = self._transform_fg(F1, F2, G1, G2)
        return self.flexion_cart.hessian(x, y, _g1, _g2, _g3, _g4, ra_0, dec_0)

    def _transform_fg(self, F1, F2, G1, G2):
        """
        Menegetti equations 2.61 2.62 definitions of flexion
        into phi_xxx, phi_xxy, phi_xyy, phi_yyy
        :param f1:
        :param f2:
        :param g1:
        :param g2:
        :return: phi_xxx, phi_xxy, phi_xyy, phi_yyy
        """
        g1 = (3*F1 + G1) * 0.5
        g2 = (3*F2 + G2) * 0.5
        g3 = (F1 - G1) * 0.5
        g4 = (F2 - G2) * 0.5
        return g1,g2,g3,g4