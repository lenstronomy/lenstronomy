
class Flexion(object):
    """
    class for flexion
    """
    param_names = ['g1', 'g2', 'g3', 'g4']
    lower_limit_default = {'g1': -0.1, 'g2': -0.1, 'g3': -0.1, 'g4': -0.1}
    upper_limit_default = {'g1': 0.1, 'g2': 0.1, 'g3': 0.1, 'g4': 0.1}

    def function(self, x, y, g1, g2, g3, g4):
        f_ = 1./6 * (g1 * x**3 + g2 * x**2 * y + g3 * x * y**2 + g4 * y**3)
        return f_

    def derivatives(self, x, y, g1, g2, g3, g4):
        f_x = 1./2.*g1*x**2 + 1./3.*g2*x*y + 1./6.*g3*y**2
        f_y = 1./6.*g2*x**2 + 1./3.*g3*x*y + 1./2.*g4*y**2
        return f_x, f_y

    def hessian(self, x, y, g1, g2, g3, g4):
        f_xx = g1*x + 1./3.*g2*y
        f_yy = 1./3.*g3*x + g4*y
        f_xy = 1./3.*g2*x + 1./3.*g3*y
        return f_xx, f_yy, f_xy
