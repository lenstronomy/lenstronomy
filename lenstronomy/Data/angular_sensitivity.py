__all__ = ["AngularSensitivity"]


class AngularSensitivity(object):
    """Telescope Angular Sensitivity class. This class provides functions describing the
    EM radiation sensitivity along different directions of some specific telescopes,
    including radio antennae of an interferometric array.

    A general reference of telescope angular sensitivity can be found in Section 5.4 of
    Bradt, H. (2004). Astronomy methods: A physical approach to astronomical
    observations. Cambridge University Press.
    """

    def __init__(self, antenna_primary_beam=None):
        """:param antenna_primary_beam: 2d numpy array.

        Primary beam is the angular power sensitivity of EM radiation antennae of some
        specific telescopes. Usually the radiation sensitivity is largest at the center
        of the Field of View (FOV), and decays as the distance increases from the
        center. If the primary beam applies, it should be multiplied on the unconvolved
        model images, as regions with less EM sensitivity get less flux from the model
        image after the multiplication with the corresponding primary beam. For
        interferometric users, the primary beam should be provided by data reduction
        softwares like CASA.
        """
        self._pb = antenna_primary_beam

    @property
    def primary_beam(self):
        """:return: 2d numpy array of primary beam."""
        return self._pb
