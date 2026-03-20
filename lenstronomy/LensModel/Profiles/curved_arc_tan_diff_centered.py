import numpy as np
from lenstronomy.LensModel.Profiles.curved_arc_tan_diff import CurvedArcTanDiff
from lenstronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LensModel.Profiles.convergence import Convergence
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import param_util

__all__ = ["CurvedArcTanDiffCentered"]


class CurvedArcTanDiffCentered(LensProfileBase):
    """Curved arc model with an additional non-zero tangential stretch differential in
    tangential direction component. The center of the lensing object is specified instead
    of the curvature radius and direction.

    Observables are:
    - center_x of the lensing object (useful to fix curvature and direction)
    - center_y of the lensing object (useful to fix curvature and direction)
    - radial stretch (plus sign) thickness of arc with parity (more generalized than the power-law slope)
    - tangential stretch (plus sign). Infinity means at critical curve
    - position of arc

    Requirements:
    - Should work with other perturbative models without breaking its meaning (say when adding additional shear terms)
    - Must best reflect the observables in lensing
    - minimal covariances between the parameters, intuitive parameterization.
    """

    param_names = [
        "tangential_stretch",
        "radial_stretch",
        "dtan_dtan",
        "center_x_lens",
        "center_y_lens",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {
        "tangential_stretch": -100,
        "radial_stretch": -5,
        "dtan_dtan": -10,
        "center_x_lens": -100,
        "center_y_lens": -100,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "tangential_stretch": 100,
        "radial_stretch": 5,
        "dtan_dtan": 10,
        "center_x_lens": 100,
        "center_y_lens": 100,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self._sie = SIE(NIE=True)
        self._mst = Convergence()
        self._curved_arc = CurvedArcTanDiff()
        super(CurvedArcTanDiffCentered, self).__init__()

    @staticmethod
    def lens_center2curvature_direction(
        tangential_stretch,
        radial_stretch,
        dtan_dtan,
        center_x_lens,
        center_y_lens,
        center_x,
        center_y,
    ):
        """
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
        :param center_x_lens: center of lensing object in image plane
        :param center_y_lens: center of lensing object in image plane
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return: tangential_stretch, radial_stretch, curvature, dtan_dtan, direction, center_x, center_y
        """
        dx = center_x - center_x_lens
        dy = center_y - center_y_lens
        curvature = 1.0 / np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)

        return (
            tangential_stretch,
            radial_stretch,
            curvature,
            dtan_dtan,
            direction,
            center_x,
            center_y,
        )

    def function(
        self,
        x,
        y,
        tangential_stretch,
        radial_stretch,
        dtan_dtan,
        center_x_lens,
        center_y_lens,
        center_x,
        center_y,
    ):
        """
        ATTENTION: there may not be a global lensing potential!

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
        :param center_x_lens: center of lensing object in image plane
        :param center_y_lens: center of lensing object in image plane
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        self._curved_arc.function(
            x,
            y,
            *self.lens_center2curvature_direction(
                tangential_stretch,
                radial_stretch,
                dtan_dtan,
                center_x_lens,
                center_y_lens,
                center_x,
                center_y,
            )
        )

    def derivatives(
        self,
        x,
        y,
        tangential_stretch,
        radial_stretch,
        dtan_dtan,
        center_x_lens,
        center_y_lens,
        center_x,
        center_y,
    ):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
        :param center_x_lens: center of lensing object in image plane
        :param center_y_lens: center of lensing object in image plane
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        self._curved_arc.derivatives(
            x,
            y,
            *self.lens_center2curvature_direction(
                tangential_stretch,
                radial_stretch,
                dtan_dtan,
                center_x_lens,
                center_y_lens,
                center_x,
                center_y,
            )
        )

    def hessian(
        self,
        x,
        y,
        tangential_stretch,
        radial_stretch,
        dtan_dtan,
        center_x_lens,
        center_y_lens,
        center_x,
        center_y,
    ):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
        :param center_x_lens: center of lensing object in image plane

        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        self._curved_arc.hessian(
            x,
            y,
            *self.lens_center2curvature_direction(
                tangential_stretch,
                radial_stretch,
                dtan_dtan,
                center_x_lens,
                center_y_lens,
                center_x,
                center_y,
            )
        )
