from lenstronomy.Sampling.Likelihoods.time_delay_likelihood import TimeDelayLikelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt
import numpy as np
import pytest
import copy


class TestImageLikelihood(object):

    def setup(self):
        pass

    def test_logL(self):

        z_source = 1.5
        z_lens = 0.5
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
        lensCosmo = LensCosmo(cosmo=cosmo, z_lens=z_lens, z_source=z_source)

        # make class instances for a chosen lens model type

        # chose a lens model
        lens_model_list = ['SPEP', 'SHEAR']
        # make instance of LensModel class
        lensModel = LensModel(lens_model_list=lens_model_list, cosmo=cosmo, z_lens=z_lens, z_source=z_source)
        # we require routines accessible in the LensModelExtensions class
        # make instance of LensEquationSolver to solve the lens equation
        lensEquationSolver = LensEquationSolver(lensModel=lensModel)

        # make choice of lens model

        # we chose a source position (in units angle)
        x_source, y_source = 0.02, 0.01
        # we chose a lens model
        kwargs_lens = [{'theta_E': 1., 'e1': 0.1, 'e2': 0.2, 'gamma': 2., 'center_x': 0, 'center_y': 0},
                       {'gamma1': 0.05, 'gamma2': -0.01}]

        # compute image positions and their (finite) magnifications

        # we solve for the image position(s) of the provided source position and lens model
        x_img, y_img = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens, sourcePos_x=x_source,
                                                                     sourcePos_y=y_source)

        point_source_list = ['LENSED_POSITION']
        kwargs_ps = [{'ra_image': x_img, 'dec_image': y_img}]
        pointSource = PointSource(point_source_type_list=point_source_list)
        t_days = lensModel.arrival_time(x_img, y_img, kwargs_lens)
        time_delays_measured = t_days[1:] - t_days[0]
        time_delays_uncertainties = np.array([0.1, 0.1, 0.1])
        self.td_likelihood = TimeDelayLikelihood(time_delays_measured, time_delays_uncertainties, lens_model_class=lensModel, point_source_class=pointSource)
        kwargs_cosmo = {'D_dt': lensCosmo.ddt}
        logL = self.td_likelihood.logL(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, kwargs_cosmo=kwargs_cosmo)
        npt.assert_almost_equal(logL, 0, decimal=8)

        time_delays_measured_new = copy.deepcopy(time_delays_measured)
        time_delays_measured_new[0] += 0.1
        td_likelihood = TimeDelayLikelihood(time_delays_measured_new, time_delays_uncertainties,
                                                 lens_model_class=lensModel, point_source_class=pointSource)
        kwargs_cosmo = {'D_dt': lensCosmo.ddt}
        logL = td_likelihood.logL(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, kwargs_cosmo=kwargs_cosmo)
        npt.assert_almost_equal(logL, -0.5, decimal=8)

    
        # Test a covariance matrix being used
        time_delays_cov = np.diag([0.1, 0.1, 0.1])**2
        td_likelihood = TimeDelayLikelihood(time_delays_measured_new, time_delays_cov,
                                                 lens_model_class=lensModel, point_source_class=pointSource)
        logL = td_likelihood.logL(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, kwargs_cosmo=kwargs_cosmo)
        npt.assert_almost_equal(logL, -0.5, decimal=8)

        # Test behaviour with a wrong number of images
        time_delays_measured_new = time_delays_measured_new[:-1]
        time_delays_uncertainties = time_delays_uncertainties[:-1] # remove last image
        td_likelihood = TimeDelayLikelihood(time_delays_measured_new, time_delays_uncertainties,
                                                 lens_model_class=lensModel, point_source_class=pointSource)
        logL = td_likelihood.logL(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, kwargs_cosmo=kwargs_cosmo)
        npt.assert_almost_equal(logL, -10**15, decimal=8)


if __name__ == '__main__':
    pytest.main()
