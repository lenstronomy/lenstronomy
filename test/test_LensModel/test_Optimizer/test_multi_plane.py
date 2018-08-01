from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
from lenstronomy.Util.util import sort_image_index
import numpy.testing as npt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import pytest
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Optimizer.multi_plane import SplitMultiplane
from lenstronomy.LensModel.lens_model import LensModel

class TestMultiPlaneOptimizer(object):

    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)

    x_pos_simple, y_pos_simple = np.array([0.69190974, -0.58959536, 0.75765166, -0.70329933]), \
                                 np.array([-0.94251661, 1.01956872, 0.45230274, -0.43988017])
    magnification_simple = [1, 0.9848458, 0.63069122, 0.54312452]

    redshift_list_simple = [0.5,0.5]
    lens_model_list_simple = ['SPEP', 'SHEAR']
    kwargs_lens_simple = [{'theta_E': 0.7, 'center_x': 0.0, 'center_y': 0, 'e1': 0.0185665252864011, 'gamma': 2.,
                           'e2': 0.08890716633399057}, {'e1': 0.00418890660015825, 'e2': -0.02908846518073248}]

    lens_model_list_subs = lens_model_list_simple + ['NFW', 'NFW', 'CONVERGENCE', 'NFW', 'NFW', 'NFW', 'CONVERGENCE', 'NFW', 'NFW', 'NFW', 'CONVERGENCE', 'NFW', 'NFW', 'NFW', 'NFW', 'CONVERGENCE', 'NFW', 'NFW', 'CONVERGENCE', 'NFW', 'NFW', 'NFW', 'NFW', 'CONVERGENCE', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW', 'NFW']
    redshift_list_subs = redshift_list_simple + [0.4, 0.4, 0.4, 0.44, 0.44, 0.44, 0.44, 0.46, 0.46, 0.46, 0.46, 0.5, 0.5, 0.52, 0.52, 0.52, 0.54, 0.54, 0.54, 0.56, 0.56, 0.56, 0.56, 0.56, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    kwargs_lens_subs = kwargs_lens_simple+[{'theta_Rs': 0.001412062596925915, 'center_y': 0.24656210487507832, 'center_x': 2.2563536452676094, 'Rs': 0.13153652610784558}, {'theta_Rs': 0.0003266721646887561, 'center_y': -2.2537377968028562, 'center_x': -1.9231242785130633, 'Rs': 0.09052504221722896}, {'kappa_ext': -0.00018652458846742356}, {'theta_Rs': 0.0002768278524857471, 'center_y': -2.5300463951767482, 'center_x': -1.1008658843496169, 'Rs': 0.10887116705229924}, {'theta_Rs': 0.0010882861827858944, 'center_y': -0.5116248584878305, 'center_x': -2.2079385997670586, 'Rs': 0.13088494450917593}, {'theta_Rs': 0.0004430997039697562, 'center_y': 0.09162780132661084, 'center_x': 2.3387071686179985, 'Rs': 0.09292434281890832}, {'kappa_ext': -0.00016086168143360287}, {'theta_Rs': 0.000943447162015366, 'center_y': 0.42631199643246565, 'center_x': -1.1785481342450466, 'Rs': 0.12663072576335713}, {'theta_Rs': 0.003679582104387777, 'center_y': -0.29936556658680563, 'center_x': -2.938753425690743, 'Rs': 0.19049041371579079}, {'theta_Rs': 0.0006476767666378457, 'center_y': -1.4323588188234009, 'center_x': 0.687346807684996, 'Rs': 0.10342523024178062}, {'kappa_ext': -0.0007659935284177056}, {'theta_Rs': 0.00019502979757386003, 'center_y': 0.08595427415912876, 'center_x': -2.658087677021249, 'Rs': 0.06040919270316341}, {'theta_Rs': 0.0023591653615883577, 'center_y': -1.4401540506955517, 'center_x': 1.2851994873009496, 'Rs': 0.13058008642852503}, {'theta_Rs': 0.0017752759204026638, 'center_y': 0.4500345648014831, 'center_x': -0.24469265869604165, 'Rs': 0.13662516805603894}, {'theta_Rs': 0.0020509540907628393, 'center_y': -0.11982069009384917, 'center_x': -2.130065697701725, 'Rs': 0.12206022829785806}, {'kappa_ext': -0.0004325247964511999}, {'theta_Rs': 0.0005658021780232771, 'center_y': -0.07772648837718947, 'center_x': -1.0635710090144175, 'Rs': 0.10800351966016286}, {'theta_Rs': 0.0004024805220970874, 'center_y': -0.2497624264662672, 'center_x': -0.4706673918278134, 'Rs': 0.09497278509635483}, {'kappa_ext': -6.524480665960749e-05}, {'theta_Rs': 0.0010358173592380955, 'center_y': -1.8134040680107917, 'center_x': -1.3361380195976233, 'Rs': 0.0931587156137893}, {'theta_Rs': 0.0009523003940647902, 'center_y': 1.803256946455538, 'center_x': -0.1849082889233431, 'Rs': 0.09855271033453286}, {'theta_Rs': 0.0035457791371316773, 'center_y': 0.5085374508824559, 'center_x': 1.2785342576406384, 'Rs': 0.21647109719977772}, {'theta_Rs': 0.0010272184584282182, 'center_y': 1.3989407374040506, 'center_x': 0.19144330036270787, 'Rs': 0.09788070190024278}, {'kappa_ext': -0.0008232792751288389}, {'theta_Rs': 0.0006246270285733823, 'center_y': 1.3687685414832078, 'center_x': -1.5959707046944127, 'Rs': 0.0903688291964993}, {'theta_Rs': 0.0012772841513627442, 'center_y': -0.22763065588454026, 'center_x': -0.6731925214311782, 'Rs': 0.12189164048227184}, {'theta_Rs': 0.0012383649170189838, 'center_y': 0.4905192454361529, 'center_x': 1.1417942926165194, 'Rs': 0.12252750782728343}, {'theta_Rs': 0.0017011961226501896, 'center_y': 2.6738577243155612, 'center_x': 0.9267064164198441, 'Rs': 0.13658232453774705}, {'theta_Rs': 0.0005797882336947124, 'center_y': 2.230485839095704, 'center_x': -0.4905230078834331, 'Rs': 0.0962801335324886}, {'theta_Rs': 0.0007625286400664437, 'center_y': -0.7077702662629382, 'center_x': 2.7339113203298075, 'Rs': 0.09552931229827409}, {'theta_Rs': 0.0015581861072900152, 'center_y': 0.17448399589498956, 'center_x': -0.8206703347011359, 'Rs': 0.11411102288379214}, {'theta_Rs': 0.0025087889149337686, 'center_y': -1.9072341892104239, 'center_x': -1.0380613066227355, 'Rs': 0.1278040282626097}, {'theta_Rs': 0.00027105058731691825, 'center_y': -2.1835418529756128, 'center_x': -0.7394736763966502, 'Rs': 0.08806857973933362}, {'theta_Rs': 0.000511360452394143, 'center_y': 1.4304950928583413, 'center_x': -1.7361402920880138, 'Rs': 0.11116701215290772}, {'theta_Rs': 0.0011090930720554857, 'center_y': 1.7569556801937074, 'center_x': -0.48068985081003185, 'Rs': 0.1096783460968247}, {'theta_Rs': 0.004776091410300207, 'center_y': -2.2241400889495195, 'center_x': -1.3986098040475345, 'Rs': 0.2531224966045142}, {'theta_Rs': 0.0004609511617975171, 'center_y': -0.8651759594769587, 'center_x': 2.2583061557021304, 'Rs': 0.09299003726735278}, {'theta_Rs': 0.0012513907509337909, 'center_y': 2.636082864564332, 'center_x': -0.8928277654358698, 'Rs': 0.11877853793320559}, {'theta_Rs': 0.0022316835203548754, 'center_y': 1.574230987801823, 'center_x': -1.0269408267368934, 'Rs': 0.10363489922593623}]

    lens_model = LensModel(lens_model_list_subs, z_source=1.5, redshift_list=redshift_list_subs, cosmo=cosmo,
                           multi_plane=True)
    solver = LensEquationSolver(lens_model)

    optimizer_simple = Optimizer(x_pos_simple, y_pos_simple, magnification_target=magnification_simple, redshift_list=redshift_list_simple,
                                 lens_model_list=lens_model_list_simple, kwargs_lens=kwargs_lens_simple, multiplane=True, verbose=True,
                                 z_source=1.5,z_main=0.5,astropy_instance=cosmo,optimizer_routine='optimize_SPEP_shear')

    optimizer_subs = Optimizer(x_pos_simple, y_pos_simple, magnification_target=magnification_simple, redshift_list=redshift_list_subs,
                               lens_model_list=lens_model_list_subs, kwargs_lens=kwargs_lens_subs, multiplane=True, verbose=True,
                               z_source=1.5,z_main=0.5,astropy_instance=cosmo,optimizer_routine='optimize_SPEP_shear')

    def test_split_multi_plane(self):

        split = SplitMultiplane(x_pos=self.x_pos_simple,y_pos=self.y_pos_simple,full_lensmodel=self.lens_model,
                                lensmodel_params=self.kwargs_lens_subs,interpolated=False,z_source=1.5,z_macro=0.5,
                                astropy_instance=self.cosmo,verbose=True)
        x_test = np.array([1,-1,0.5,0.25])
        y_test = np.array([1,-1,0.6,-0.3])
        betax_true,betay_true = split.ray_shooting_full(x_test,y_test)
        magnification_true = split.magnification_full(self.x_pos_simple,self.y_pos_simple)

        betax_split,betay_split = split.ray_shooting(split.macro_args)
        magnification_split = split.magnification(split.macro_args)

        betax_true = np.mean(betax_true)
        betay_true = np.mean(betay_true)
        betax_split = np.mean(betax_split)
        betay_split = np.mean(betay_split)

        npt.assert_(split.macro_args==self.kwargs_lens_subs[0:2])
        npt.assert_(split.macromodel_lensmodel.redshift_list == self.redshift_list_subs[0:2])
        npt.assert_(split.macromodel_lensmodel.lens_model_list == self.lens_model_list_subs[0:2])

        npt.assert_(split.halos_args == self.kwargs_lens_subs[2:])
        npt.assert_(split.halos_lensmodel.redshift_list == self.redshift_list_subs[2:])
        npt.assert_(split.halos_lensmodel.lens_model_list == self.lens_model_list_subs[2:])

        count = 0
        for i,z in enumerate(self.redshift_list_subs):
            if z > 0.5:
                count+=1
        npt.assert_(len(split.background_args) == count)

        npt.assert_almost_equal(betax_true,betax_split,decimal=3)
        npt.assert_almost_equal(betay_true, betay_split,decimal=3)
        npt.assert_almost_equal(magnification_true, magnification_split,decimal=1)

    def test_multi_plane_simple(self):
        """

        :param tol: image position tolerance
        :return:
        """
        kwargs_lens, source, [x_image,y_image] = self.optimizer_simple.optimize(n_particles=50,n_iterations=200, restart = 2)
        index = sort_image_index(x_image, y_image, self.x_pos_simple, self.y_pos_simple)

        x_image = x_image[index]
        y_image = y_image[index]
        mags = self.optimizer_simple.optimizer_amoeba.lensModel.magnification(x_image, y_image, kwargs_lens)
        mags = np.absolute(mags)
        mags *= max(mags) ** -1

        npt.assert_almost_equal(x_image, self.x_pos_simple, decimal=3)
        npt.assert_almost_equal(y_image, self.y_pos_simple, decimal=3)
        npt.assert_array_less(np.absolute(self.magnification_simple - mags) * 0.2 ** -1, [1, 1, 1, 1])

    def test_multi_plane_subs(self,tol=0.004):
        """
        Should be a near perfect fit since the LOS model is the same used to create the data.
        :return:
        """
        kwargs_lens, source, [x_image,y_image] = self.optimizer_subs.optimize(n_particles=50,n_iterations=200, restart = 2)

        index = sort_image_index(x_image, y_image, self.x_pos_simple, self.y_pos_simple)
        x_image = x_image[index]
        y_image = y_image[index]

        mags = self.optimizer_subs.optimizer_amoeba.lensModel.magnification(x_image, y_image, kwargs_lens)
        mags = np.absolute(mags)
        mags *= max(mags) ** -1

        dx = np.absolute(x_image - self.x_pos_simple)
        dy = np.absolute(y_image - self.y_pos_simple)

        npt.assert_array_less(dx, [tol] * len(dx))
        npt.assert_array_less(dy, [tol] * len(dy))
        npt.assert_array_less(np.absolute(self.magnification_simple - mags) * 0.2 ** -1, [1, 1, 1, 1])

f = TestMultiPlaneOptimizer()
f.test_split_multi_plane()
#f.test_multi_plane_simple()
f.test_multi_plane_subs()

exit(1)
f.test_
if __name__ == '__main__':
    pytest.main()

