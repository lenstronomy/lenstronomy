from lenstronomy.LightModel.Profiles.power_law import PowerLaw


class Chameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical double power-law profile

    """
    def __init__(self):
        self.power_law = PowerLaw()

    def function(self, x, y, amp, flux_ratio, gamma1, gamma2, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param amp: amplitude of first power-law flux
        :param flux_ratio: ratio of amplitudes of first to second power-law profile
        :param gamma1: power-law slope
        :param gamma2: power-law slope
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: center
        :param center_y: center
        :return: flux of chameleon profile
        """
        flux1 = self.power_law.function(x, y, 1, gamma1, e1, e2, center_x, center_y)
        flux2 = self.power_law.function(x, y, 1, gamma2, e1, e2, center_x, center_y)
        flux = amp * (flux1 + flux_ratio * flux2)
        return flux

    def light_3d(self, r, amp, flux_ratio, gamma1, gamma2, e1, e2):
        """

        :param r:
        :param amp:
        :param flux_ratio:
        :param gamma1:
        :param gamma2:
        :param e1:
        :param e2:
        :return:
        """
        light1 = self.power_law.light_3d(r, amp, gamma1, e1, e2)
        light2 = self.power_law.light_3d(r, amp*flux_ratio, gamma2, e1, e2)
        light = light1 + light2
        return light