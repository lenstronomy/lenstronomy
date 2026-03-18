__author__ = "Guanhua Rui,  Wei Du"

import numpy as np
import pytest
import numpy.testing as npt

import lenstronomy.Util.param_util as param_util
from lenstronomy.LightModel.Profiles.pl_sersic import PL_Sersic


class TestPLSersic(object):
    """Tests the PL_Sersic methods."""

    def setup_method(self):
        self.pl_sersic = PL_Sersic()

    def test_pl_sersic(self):
        x = np.array([1.0])
        y = np.array([2.0])

        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0
        center_x, center_y = 0.0, 0.0

        values = self.pl_sersic.function(
            x,
            y,
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=center_x,
            center_y=center_y,
        )
        npt.assert_almost_equal(values[0], 0.07440080198329953, decimal=10)

    def test_pl_sersic_array(self):
        x = np.array([2.0, 3.0, 4.0])
        y = np.array([1.0, 1.0, 1.0])

        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0

        values = self.pl_sersic.function(
            x,
            y,
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        npt.assert_almost_equal(
            values,
            np.array(
                [0.07440080198329953, 0.011670302367673888, 0.0017081167108147768]
            ),
            decimal=10,
        )

    def test_pl_sersic_center(self):
        x = np.array([0.0])
        y = np.array([0.0])

        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0

        values = self.pl_sersic.function(
            x,
            y,
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        npt.assert_almost_equal(values[0], 3.620310212856137, decimal=10)

    def test_pl_sersic_inner_branch(self):
        # R < r_c branch (includes hypergeometric term + integral)
        x = np.array([0.1])
        y = np.array([0.2])

        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0

        values = self.pl_sersic.function(
            x,
            y,
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        npt.assert_almost_equal(values[0], 2.6456088942857, decimal=10)

    def test_pl_sersic_cutoff(self):
        # cutoff: I(R)=0 for R > max_R_frac * R_sersic
        x = np.array([2000.0])
        y = np.array([0.0])

        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0

        values = self.pl_sersic.function(
            x,
            y,
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
            max_R_frac=1000.0,
        )
        npt.assert_almost_equal(values[0], 0.0, decimal=12)

    def test_pl_sersic_elliptic(self):
        x = np.array([1.0])
        y = np.array([2.0])

        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7

        phi_G, q = 1.0, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        values = self.pl_sersic.function(
            x,
            y,
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        npt.assert_almost_equal(values[0], 0.0930628278044294, decimal=10)

        values_center = self.pl_sersic.function(
            np.array([0.0]),
            np.array([0.0]),
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        npt.assert_almost_equal(values_center[0], 3.620310212856137, decimal=10)

    def test_amp_linearity_outer(self):
        # choose an outer-branch point (R>r_c) to avoid expensive inner-branch integral
        x = np.array([1.0])
        y = np.array([2.0])

        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0

        v1 = self.pl_sersic.function(
            x,
            y,
            1.0,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        v2 = self.pl_sersic.function(
            x,
            y,
            2.0,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        npt.assert_almost_equal(v2[0] / v1[0], 2.0, decimal=10)

    def test_pl_sersic_cutoff_scalar(self):
        # cover scalar branch: np.ndim(R)==0 and R > R_max -> return 0.0
        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0

        val = self.pl_sersic.function(
            2000.0,  # scalar x
            0.0,  # scalar y
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
            max_R_frac=1000.0,
        )
        npt.assert_almost_equal(val, 0.0, decimal=12)

    def test_pl_sersic_inner_branch_scalar(self):
        # cover scalar branch: np.ndim(R)==0 and R < r_c -> I_inner_scalar(float(R))
        amp = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        alpha_c = 0.5
        r_c = 0.7
        e1, e2 = 0.0, 0.0

        # scalar call
        v_scalar = self.pl_sersic.function(
            0.1,
            0.2,
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )

        # compare against the already-tested array pathway for the same point
        v_array = self.pl_sersic.function(
            np.array([0.1]),
            np.array([0.2]),
            amp,
            R_sersic,
            n_sersic,
            alpha_c,
            r_c,
            e1,
            e2,
            center_x=0.0,
            center_y=0.0,
        )
        npt.assert_almost_equal(v_scalar, float(v_array[0]), decimal=12)

    def test_total_flux_alpha_c_ge_3_raises(self):
        # cover total_flux: alpha_c >= 3 -> raise ValueError
        with pytest.raises(ValueError, match="alpha_c must be < 3"):
            self.pl_sersic.total_flux(
                amp=1.0,
                R_sersic=1.0,
                n_sersic=1.0,
                alpha_c=3.0,
                r_c=0.7,
                e1=0.0,
                e2=0.0,
            )


if __name__ == "__main__":
    pytest.main()
