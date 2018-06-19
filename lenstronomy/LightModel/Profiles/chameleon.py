class Chameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical double power-law profile

    """

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
        return 0