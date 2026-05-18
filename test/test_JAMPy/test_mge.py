import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.JAMPy.mge import MGEMass, MGELight
from lenstronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LightModel.Profiles.hernquist import Hernquist
from lenstronomy.LightModel.Profiles.sersic import Sersic
from lenstronomy.Analysis.light2mass import light2mass_interpol


class TestMGEMass:
    def setup_method(self):
        self.kw_sie = {
            "theta_E": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": 0.01,
        }
        self.kw_gauss = {"sigma": 1.1, "amp": 1.0, "center_x": 0.0, "center_y": 0.0}

    def test_radial_convergence(self):
        mge_mass = MGEMass(["SIE", "GAUSSIAN"])
        kwargs_list = [self.kw_sie, self.kw_gauss]
        r = np.array([0.1, 0.5, 1.0])
        dens = mge_mass.radial_convergence(r, kwargs_list)
        npt.assert_almost_equal(dens, np.array([5.1453, 1.12148, 0.58844]), decimal=3)

    def test_einstein_radius(self):
        mge_mass = MGEMass(["SIE"])
        theta = 2.5
        assert mge_mass.einstein_radius([{"theta_E": theta}]) == theta

        mp2 = MGEMass(["SIE", "GAUSSIAN"])
        kwargs_list = [self.kw_sie, self.kw_gauss]
        theta_composite = mp2.einstein_radius(kwargs_list)
        npt.assert_almost_equal(theta_composite, 1.12273, decimal=3)

    def test_mge_mass(self):
        mge_mass = MGEMass(["SIE"])
        r_test = np.logspace(  # this must be in logspace
            np.log10(1e-2),
            np.log10(100),
            300,
        )
        surf_mass, sigma_mass = mge_mass.mge_fit([self.kw_sie])
        mge_surf_1d = _mge(r_test, surf_mass, sigma_mass)
        theta_E = self.kw_sie["theta_E"]
        rho0 = SIE.theta2rho(theta_E)
        sie_surf_1d = SIE.density_2d(r_test, 0, rho0)
        npt.assert_allclose(mge_surf_1d, sie_surf_1d, rtol=0.1)

    def test_mge_mass_mge_prof(self):
        mge_mass = MGEMass(["MULTI_GAUSSIAN"])
        kw_mge = {"amp": np.arange(5), "sigma": np.arange(1, 6)}
        surf_mass, sigma_mass = mge_mass.mge_fit(
            [kw_mge],
        )
        npt.assert_allclose(surf_mass, np.arange(1, 5), rtol=1e-2)
        npt.assert_allclose(sigma_mass, np.arange(2, 6), rtol=1e-2)

    def test_mge_interpolated_mass(self):
        r_test = np.logspace(  # this must be in logspace
            np.log10(0.1),
            np.log10(2),
            100,
        )
        mge_interp = MGEMass(["INTERPOL"])
        mge_gauss = MGEMass(["GAUSSIAN"])
        kw_interp = light2mass_interpol(
            ["GAUSSIAN"],
            kwargs_lens_light=[self.kw_gauss],
            numPix=200,
            deltaPix=0.2,
            subgrid_res=3,
        )
        surf_interp = mge_interp.radial_convergence(
            r_test,
            [kw_interp],
        )
        surf_gauss = mge_gauss.radial_convergence(
            r_test,
            [self.kw_gauss],
        )
        npt.assert_allclose(
            surf_interp / surf_interp[0], surf_gauss / surf_gauss[0], rtol=0.1
        )

    def test_parse_kwargs(self):
        mge_mass = MGEMass(["SIE", "GAUSSIAN"])
        kw_sie = self.kw_sie.copy()
        kw_sie.pop("e1")
        kw_sie.pop("e2")
        kw_gauss = self.kw_gauss | {"e1": 0, "e2": 0}
        kwargs_list = [kw_sie, kw_gauss]
        kwargs_list_parsed = mge_mass._parse_kwargs(kwargs_list)
        assert "e1" in kwargs_list_parsed[0]
        assert "e2" in kwargs_list_parsed[0]
        assert "e1" not in kwargs_list_parsed[1]
        assert "e2" not in kwargs_list_parsed[1]


class TestMGELight:
    def setup_method(self):
        self.kw_sersic = {
            "R_sersic": 1.5,
            "amp": 1.0,
            "n_sersic": 2.5,
            "e1": 0.1,
            "e2": 0.01,
            "center_x": 0.2,
            "center_y": -0.1,
        }
        self.kw_hernquist = {"Rs": 0.8, "amp": 1.0, "center_x": 0.2, "center_y": -0.1}
        self.kw_gaussian = {"sigma": 0.5, "amp": 1.0}

    def test_radial_surface_brightness(self):
        mge_l = MGELight(["SERSIC", "GAUSSIAN"])
        kwargs_list = [
            dict(
                self.kw_sersic,
                **{"center_x": 0.1, "center_y": -0.1, "e1": 0.1, "e2": -0.1}
            ),
            dict(self.kw_gaussian, **{"center_x": 0.1, "center_y": -0.1}),
        ]
        r = np.array([0, 1, 2])
        val = mge_l.radial_surface_brightness(r, kwargs_list)
        npt.assert_almost_equal(val, np.array([97.2906, 2.0985, 0.5659]), decimal=4)

    def test_effective_radius(self):
        mge_l_sersic = MGELight(["SERSIC"])
        assert (
            mge_l_sersic.effective_radius([self.kw_sersic])
            == self.kw_sersic["R_sersic"]
        )
        mge_l_hern = MGELight(["HERNQUIST"])
        npt.assert_almost_equal(
            mge_l_hern.effective_radius([self.kw_hernquist]),
            1.8153 * self.kw_hernquist["Rs"],
            decimal=4,
        )
        mge_l = MGELight(["SERSIC", "HERNQUIST"])
        kwargs_list = [
            self.kw_sersic,
            self.kw_hernquist,
        ]
        r_eff = mge_l.effective_radius(kwargs_list)
        npt.assert_almost_equal(r_eff, 0.849914, decimal=3)

    def test_mge_lum_hernquist(self):
        mge_l = MGELight(["HERNQUIST"])
        r_test = np.logspace(  # this must be in logspace
            np.log10(1e-2),
            np.log10(100),
            300,
        )
        surf_lum, sigma_lum = mge_l.mge_fit([self.kw_hernquist])
        mge_surf_1d = _mge(r_test, surf_lum, sigma_lum)
        hernq = Hernquist()
        hernq_surf_1d = hernq.function(
            x=r_test, y=0, Rs=self.kw_hernquist["Rs"], amp=self.kw_hernquist["amp"]
        )
        npt.assert_allclose(mge_surf_1d, hernq_surf_1d, rtol=0.1, atol=1e-5)

    def test_mge_lum_sersic(self):
        mge_l = MGELight(["SERSIC"])
        r_test = np.logspace(  # this must be in logspace
            np.log10(1e-2),
            np.log10(100),
            300,
        )
        surf_lum, sigma_lum = mge_l.mge_fit([self.kw_sersic])
        mge_surf_1d = _mge(r_test, surf_lum, sigma_lum)
        sersic = Sersic()
        sersic_surf_1d = sersic.function(
            x=r_test,
            y=0,
            n_sersic=self.kw_sersic["n_sersic"],
            R_sersic=self.kw_sersic["R_sersic"],
            amp=self.kw_sersic["amp"],
        )
        npt.assert_allclose(mge_surf_1d, sersic_surf_1d, rtol=0.1, atol=1e-5)

    def test_mge_lum_mge_prof(self):
        mge_l = MGELight(["MULTI_GAUSSIAN"])
        kw_mge = {"amp": np.arange(5), "sigma": np.arange(1, 6)}

        surf_lum, sigma_lum = mge_l.mge_fit(
            [kw_mge],
        )
        # should remove the zero amplitude
        npt.assert_allclose(surf_lum, np.arange(1, 5), rtol=1e-3)
        npt.assert_allclose(sigma_lum, np.arange(2, 6), rtol=1e-3)

    def test_parse_kwargs(self):
        mge_l = MGELight(["SERSIC_ELLIPSE", "GAUSSIAN"])
        kw_sersic = self.kw_sersic.copy()
        kw_sersic.pop("e1")
        kw_sersic.pop("e2")
        kw_gauss = self.kw_gaussian | {"e1": 0, "e2": 0}
        kwargs_list = [kw_sersic, kw_gauss]
        kwargs_list_parsed = mge_l._parse_kwargs(kwargs_list)
        assert "e1" in kwargs_list_parsed[0]
        assert "e2" in kwargs_list_parsed[0]
        assert "e1" not in kwargs_list_parsed[1]
        assert "e2" not in kwargs_list_parsed[1]


class TestRaise:
    def test_invalid_profile_name_raises(self):
        with pytest.raises(ValueError):
            MGEMass(["INVALID_PROFILE_NAME"])

        with pytest.raises(ValueError):
            MGELight(["INVALID_PROFILE_NAME"])


def _gaussian(r, amp, sigma):
    return amp * np.exp(-0.5 * (r / sigma) ** 2)


def _mge(r, amps, sigmas):
    amps /= 2 * np.pi * sigmas**2
    total = np.zeros_like(r)
    for amp, sigma in zip(amps, sigmas):
        total += _gaussian(r, amp, sigma)
    return total


if __name__ == "__main__":
    pytest.main()
