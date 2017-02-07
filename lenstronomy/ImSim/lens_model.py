__author__ = 'sibirrer'

#file which contains class for lens model routines


class LensModel(object):

    def __init__(self, kwargs_options):
        from astrofunc.LensingProfiles.external_shear import ExternalShear
        ellipse_type = kwargs_options.get('ellipse_type', 'SPEP')
        self.shear = ExternalShear()
        if kwargs_options['lens_type'] == 'GAUSSIAN':
            from astrofunc.LensingProfiles.gaussian import Gaussian
            self.func = Gaussian()
        elif kwargs_options['lens_type'] == 'SIS':
            from astrofunc.LensingProfiles.sis import SIS
            self.func = SIS()
        elif kwargs_options['lens_type'] == 'SPP':
            from astrofunc.LensingProfiles.spp import SPP
            self.func = SPP()
        elif kwargs_options['lens_type'] == 'SPEP':
            from astrofunc.LensingProfiles.spep import SPEP
            self.func = SPEP()
        elif kwargs_options['lens_type'] == 'SPEMD':
            from astrofunc.LensingProfiles.spemd import SPEMD
            self.func = SPEMD()
        elif kwargs_options['lens_type'] == 'ELLIPSE':
            from astrofunc.LensingProfiles.ellipse import Ellipse
            self.func = Ellipse(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'SPEP_SIS':
            from astrofunc.LensingProfiles.spep_sis import SPEP_SIS
            self.func = SPEP_SIS(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'SPEP_SPP':
            from astrofunc.LensingProfiles.spep_spp import SPEP_SPP
            self.func = SPEP_SPP(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'INTERPOL':
            from astrofunc.LensingProfiles.interpol import Interpol_func
            self.func = Interpol_func()
        elif kwargs_options['lens_type'] == 'SHAPELETS_POLAR':
            from astrofunc.LensingProfiles.shapelet_pot import PolarShapelets
            self.func = PolarShapelets()
        elif kwargs_options['lens_type'] == 'SHAPELETS_CART':
            from astrofunc.LensingProfiles.shapelet_pot_2 import CartShapelets
            self.func = CartShapelets()
        elif kwargs_options['lens_type'] == 'SPEP_SHAPELETS':
            from astrofunc.LensingProfiles.spep_shapelets import SPEP_Shapelets
            self.func = SPEP_Shapelets(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'SPEP_SPP_SHAPELETS':
            from astrofunc.LensingProfiles.spep_shapelets import SPEP_SPP_Shapelets
            self.func = SPEP_SPP_Shapelets(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE':
            from astrofunc.LensingProfiles.spep_spp_dipole import SPEP_SPP_Dipole
            self.func = SPEP_SPP_Dipole(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE_SHAPELETS':
            from astrofunc.LensingProfiles.spep_spp_dipole import SPEP_SPP_Dipole_Shapelets
            self.func = SPEP_SPP_Dipole_Shapelets(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'SPEP_NFW':
            from astrofunc.LensingProfiles.spep_nfw import SPEP_NFW
            self.func = SPEP_NFW(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'NFW':
            from astrofunc.LensingProfiles.nfw import NFW
            self.func = NFW()
        elif kwargs_options['lens_type'] == 'DIPOLE':
            from astrofunc.LensingProfiles.dipole import Dipole
            self.func = Dipole()
        elif kwargs_options['lens_type'] == 'SPEP_SPP_DIPOLE':
            from astrofunc.LensingProfiles.spep_spp_dipole import SPEP_SPP_Dipole
            self.func = SPEP_SPP_Dipole(type=ellipse_type)
        elif kwargs_options['lens_type'] == 'EXTERNAL_SHEAR':
            from astrofunc.LensingProfiles.external_shear import ExternalShear
            self.func = ExternalShear()
        elif kwargs_options['lens_type'] == 'NONE':
            from astrofunc.LensingProfiles.no_lens import NoLens
            self.func = NoLens()
        else:
            raise ValueError('options do not include a valid lens model!', kwargs_options['lens_type'])
        self.external_shear = kwargs_options.get('external_shear', False)
        self.foreground_shear = kwargs_options.get('foreground_shear', False)
        self.add_clump = kwargs_options.get('add_clump', False)
        self.clump_type = kwargs_options.get('clump_type', 'SIS_TRUNCATED')
        if self.clump_type == 'SIS_TRUNCATED':
            from astrofunc.LensingProfiles.sis_truncate import SIS_truncate
            self.clump = SIS_truncate()
        elif self.clump_type == 'NFW':
            from astrofunc.LensingProfiles.nfw import NFW, HaloParam
            self.clump = NFW()
        else:
            raise ValueError('clump_type %s not valid!' % self.clump_type)
        self.perturb_alpha = kwargs_options.get("perturb_alpha", False)
        if self.perturb_alpha:
            self.alpha_perturb_x = kwargs_options["alpha_perturb_x"]
            self.alpha_perturb_y = kwargs_options["alpha_perturb_y"]

    def mass(self, x, y, sigma_crit, **kwargs):
        kappa = self.kappa(x, y, **kwargs)
        mass = sigma_crit*kappa
        return mass

    def potential(self, x, y, kwargs_else=None, **kwargs):
        potential = self.func.function(x, y, **kwargs)
        if self.add_clump:
            if self.clump_type == 'SIS_TRUNCATED':
                pot_clump = self.clump.function(x, y, theta_E_trunc=kwargs_else['theta_E_clump'], r_trunc=kwargs_else['r_trunc'], center_x_trunc=kwargs_else['x_clump'], center_y_trunc=kwargs_else['y_clump'])
            elif self.clump_type == 'NFW':
                pot_clump = self.clump.function(x, y, Rs=kwargs_else['r_trunc'], rho0=kwargs_else['phi_E_clump'], center_x_nfw=kwargs_else['x_clump'], center_y_nfw=kwargs_else['y_clump'], angle=True)
            potential += pot_clump
        return potential

    def alpha(self, x, y, kwargs_else=None, **kwargs):
        """
        a = grad(phi)
        """
        if self.foreground_shear and self.external_shear:
            f_x_shear1, f_y_shear1 = self.shear.derivatives(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
            x_ = x - f_x_shear1
            y_ = y - f_y_shear1
        else:
            f_x_shear1, f_y_shear1 = 0, 0
            x_ = x
            y_ = y
        f_x, f_y = self.func.derivatives(x_, y_, **kwargs)
        #f_x += f_x_shear1
        #f_y += f_y_shear1
        if self.external_shear:
            f_x_shear, f_y_shear = self.shear.derivatives(x, y, e1=kwargs_else['gamma1'], e2=kwargs_else['gamma2'])
            f_x += f_x_shear
            f_y += f_y_shear
        if self.add_clump:
            if self.clump_type == 'SIS_TRUNCATED':
                f_x_clump, f_y_clump = self.clump.derivatives(x_, y_, phi_E_trunc=kwargs_else['phi_E_clump'], r_trunc=kwargs_else['r_trunc'], center_x_trunc=kwargs_else['x_clump'], center_y_trunc=kwargs_else['y_clump'])
            elif self.clump_type == 'NFW':
                 f_x_clump, f_y_clump = self.clump.derivatives(x_, y_, Rs=kwargs_else['r_trunc'], rho0=kwargs_else['phi_E_clump'], center_x_nfw=kwargs_else['x_clump'], center_y_nfw=kwargs_else['y_clump'], angle=True)
            else:
                raise ValueError("clump type not valid!")
            f_x += f_x_clump
            f_y += f_y_clump
        alpha1 = f_x  # attention on units
        alpha2 = f_y  # attention on units
        if self.perturb_alpha:
            alpha1 += self.alpha_perturb_x
            alpha2 += self.alpha_perturb_y
        return alpha1, alpha2

    def kappa(self, x, y, kwargs_else=None, **kwargs):
        """
        k = 1/2 laplacian(phi)
        """
        f_xx, f_yy, f_xy = self.func.hessian(x, y, **kwargs)
        if self.add_clump and kwargs_else is not None:
            if self.clump_type == 'SIS_TRUNCATED':
                f_xx_clump, f_yy_clump, f_xy_clump = self.clump.hessian(x, y, phi_E_trunc=kwargs_else['phi_E_clump'], r_trunc=kwargs_else['r_trunc'], center_x_trunc=kwargs_else['x_clump'], center_y_trunc=kwargs_else['y_clump'])
            elif self.clump_type == 'NFW':
                f_xx_clump, f_yy_clump, f_xy_clump = self.clump.hessian(x, y, Rs=kwargs_else['r_trunc'], rho0=kwargs_else['phi_E_clump'], center_x_nfw=kwargs_else['x_clump'], center_y_nfw=kwargs_else['y_clump'], angle=True)
            f_xx += f_xx_clump
            f_yy += f_yy_clump
            f_xy += f_xy_clump
        kappa = 1./2 * (f_xx + f_yy)  # attention on units
        return kappa

    def gamma(self, x, y, kwargs_else=None, **kwargs):
        """
        g1 = 1/2(d^2phi/dx^2 - d^2phi/dy^2)
        g2 = d^2phi/dxdy
        """
        if self.foreground_shear and self.external_shear:
            f_x_shear1, f_y_shear1 = self.shear.derivatives(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
            x_ = x - f_x_shear1
            y_ = y - f_y_shear1
            f_xx_shear1, f_yy_shear1, f_xy_shear1 = self.shear.hessian(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
        else:
            f_xx_shear1, f_yy_shear1, f_xy_shear1 = 0, 0, 0
            x_ = x
            y_ = y

        f_xx, f_yy, f_xy = self.func.hessian(x_, y_, **kwargs)
        #f_xx += f_xx_shear1
        #f_yy += f_yy_shear1
        #f_xy += f_xy_shear1
        if self.external_shear:
            f_xx_shear, f_yy_shear, f_xy_shear = self.shear.hessian(x_, y_, e1=kwargs_else['gamma1'], e2=kwargs_else['gamma2'])
            f_xx += f_xx_shear
            f_yy += f_yy_shear
            f_xy += f_xy_shear
        if self.add_clump:
            if self.clump_type == 'SIS_TRUNCATED':
                f_xx_clump, f_yy_clump, f_xy_clump = self.clump.hessian(x_, y_, phi_E_trunc=kwargs_else['phi_E_clump'], r_trunc=kwargs_else['r_trunc'], center_x_trunc=kwargs_else['x_clump'], center_y_trunc=kwargs_else['y_clump'])
            elif self.clump_type == 'NFW':
                 f_xx_clump, f_yy_clump, f_xy_clump = self.clump.hessian(x_, y_, Rs=kwargs_else['r_trunc'], rho0=kwargs_else['phi_E_clump'], center_x_nfw=kwargs_else['x_clump'], center_y_nfw=kwargs_else['y_clump'], angle=True)
            else:
                raise ValueError("clump type not valid.")
            f_xx += f_xx_clump
            f_yy += f_yy_clump
            f_xy += f_xy_clump
        gamma1 = 1./2 * (f_xx - f_yy)  # attention on units
        gamma2 = f_xy  # attention on units
        return gamma1, gamma2

    def magnification(self, x, y, kwargs_else=None, **kwargs):
        """
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij
        """
        if self.foreground_shear and self.external_shear:
            f_x_shear1, f_y_shear1 = self.shear.derivatives(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
            x_ = x - f_x_shear1
            y_ = y - f_y_shear1
            f_xx_shear1, f_yy_shear1, f_xy_shear1 = self.shear.hessian(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
        else:
            f_xx_shear1, f_yy_shear1, f_xy_shear1 = 0, 0, 0
            x_ = x
            y_ = y

        f_xx, f_yy, f_xy = self.func.hessian(x_, y_, **kwargs)
        #f_xx += f_xx_shear1
        #f_yy += f_yy_shear1
        #f_xy += f_xy_shear1
        if self.external_shear:
            f_xx_shear, f_yy_shear, f_xy_shear = self.shear.hessian(x, y, e1=kwargs_else['gamma1'], e2=kwargs_else['gamma2'])
            f_xx += f_xx_shear
            f_yy += f_yy_shear
            f_xy += f_xy_shear
        if self.add_clump:
            if self.clump_type == 'SIS_TRUNCATED':
                f_xx_clump, f_yy_clump, f_xy_clump = self.clump.hessian(x_, y_, phi_E_trunc=kwargs_else['phi_E_clump'], r_trunc=kwargs_else['r_trunc'], center_x_trunc=kwargs_else['x_clump'], center_y_trunc=kwargs_else['y_clump'])
            elif self.clump_type == 'NFW':
                 f_xx_clump, f_yy_clump, f_xy_clump = self.clump.hessian(x_, y_, Rs=kwargs_else['r_trunc'], rho0=kwargs_else['phi_E_clump'], center_x_nfw=kwargs_else['x_clump'], center_y_nfw=kwargs_else['y_clump'], angle=True)
            else:
                raise ValueError("clump type not valid.")
            f_xx += f_xx_clump
            f_yy += f_yy_clump
            f_xy += f_xy_clump
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_xy  # attention, only works in right units of critical density
        return 1./det_A  # attention, if dividing to zero

    def all(self, x, y, kwargs_else=None, **kwargs):
        """
        specially build to reduce computational costs
        """
        if self.foreground_shear and self.external_shear:
            f_x_shear1, f_y_shear1 = self.shear.derivatives(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
            x_ = x - f_x_shear1
            y_ = y - f_y_shear1
            f_xx_shear1, f_yy_shear1, f_xy_shear1 = self.shear.hessian(x, y, e1=kwargs_else['gamma1_foreground'], e2=kwargs_else['gamma2_foreground'])
        else:
            f_x_shear1, f_y_shear1, f_xx_shear1, f_yy_shear1, f_xy_shear1 = 0, 0, 0, 0, 0
            x_ = x
            y_ = y
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.func.all(x_, y_, **kwargs)
        #f_x += f_x_shear1
        #f_y += f_y_shear1
        #f_xx += f_xx_shear1
        #f_yy += f_yy_shear1
        #f_xy += f_xy_shear1
        if self.external_shear:
            f_shear, f_x_shear, f_y_shear, f_xx_shear, f_yy_shear, f_xy_shear = self.shear.all(x, y, e1=kwargs_else['gamma1'], e2=kwargs_else['gamma2'])
            f_x += f_x_shear
            f_y += f_y_shear
            f_xx += f_xx_shear
            f_yy += f_yy_shear
            f_xy += f_xy_shear
        if self.add_clump:
            if self.clump_type == 'SIS_TRUNCATED':
                f_clump, f_x_clump, f_y_clump, f_xx_clump, f_yy_clump, f_xy_clump = self.clump.all(x_, y_, phi_E_trunc=kwargs_else['phi_E_clump'], r_trunc=kwargs_else['r_trunc'], center_x_trunc=kwargs_else['x_clump'], center_y_trunc=kwargs_else['y_clump'])
            elif self.clump_type == 'NFW':
                 f_clump, f_x_clump, f_y_clump, f_xx_clump, f_yy_clump, f_xy_clump = self.clump.all(x_, y_, Rs=kwargs_else['r_trunc'], rho0=kwargs_else['phi_E_clump'], center_x_nfw=kwargs_else['x_clump'], center_y_nfw=kwargs_else['y_clump'], angle=True)
            else:
                raise ValueError("clump type not valid.")
            f_ += f_clump
            f_x += f_x_clump
            f_y += f_y_clump
            f_xx += f_xx_clump
            f_yy += f_yy_clump
            f_xy += f_xy_clump
        potential = f_
        alpha1 = f_x  # attention on units
        alpha2 = f_y  # attention on units
        if self.perturb_alpha:
            alpha1 += self.alpha_perturb_x
            alpha2 += self.alpha_perturb_y
        kappa = 1./2 * (f_xx + f_yy)  # attention on units
        gamma1 = 1./2 * (f_xx - f_yy)  # attention on units
        gamma2 = f_xy  # attention on units
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_xy  # attention, only works in right units of critical density
        mag = 1./det_A
        return potential, alpha1, alpha2, kappa, gamma1, gamma2, mag