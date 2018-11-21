
class Flexion(object):
    """
    class for flexion
    """
    param_names = ['g1', 'g2', 'g3', 'g4', 'ra_0', 'dec_0']
    lower_limit_default = {'g1': -0.1, 'g2': -0.1, 'g3': -0.1, 'g4': -0.1, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'g1': 0.1, 'g2': 0.1, 'g3': 0.1, 'g4': 0.1, 'ra_0': 100, 'dec_0': 100}

    def function(self, x, y, g1, g2, g3, g4, ra_0=0, dec_0=0):
        x_ = x - ra_0
        y_ = y - dec_0
        f_ = 1./6 * (g1 * x_**3 + g2 * x_**2 * y_ + g3 * x_ * y_**2 + g4 * y_**3)
        return f_

    def derivatives(self, x, y, g1, g2, g3, g4, ra_0=0, dec_0=0):
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = 1./2.*g1*x_**2 + 1./3.*g2*x_*y_ + 1./6.*g3*y_**2
        f_y = 1./6.*g2*x_**2 + 1./3.*g3*x_*y_ + 1./2.*g4*y_**2
        return f_x, f_y

    def hessian(self, x, y, g1, g2, g3, g4, ra_0=0, dec_0=0):
        x_ = x - ra_0
        y_ = y - dec_0
        f_xx = g1*x_ + 1./3.*g2*y_
        f_yy = 1./3.*g3*x_ + g4*y_
        f_xy = 1./3.*g2*x_ + 1./3.*g3*y_
        return f_xx, f_yy, f_xy
