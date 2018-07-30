from lenstronomy.LensModel.Optimizer.optimizer import Optimizer
from lenstronomy.Util.util import sort_image_index
import numpy.testing as npt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import pytest

class TestMultiPlaneOptimizer(object):

    # this data was created with a full LOS realization
    x_pos = [-0.51087,0.14855,-1.05363,0.90641]
    y_pos = [0.96657,1.03966,0.0905,-1.21829]
    x_pos, y_pos = np.array(x_pos), np.array(y_pos)
    magnification = [1,0.63039,0.353034,0.196263]
    cosmo = FlatLambdaCDM(H0=70,Om0=0.29)

    # 'simple' lens model is not the same as what is used to create data
    redshift_list_simple = [0.5, 0.5]
    lens_model_list_simple = ['SPEMD', 'SHEAR']
    kwargs_lens_simple = [{'theta_E': 0.7, 'center_x': 0.01, 'center_y': 0, 'e1': 0.0585665252864011, 'gamma': 2,
                           'e2': 0.08890716633399057}, {'e1': 0.00418890660015825, 'e2': -0.05908846518073248}]

    # this is the lens model used to create the data
    redshift_list_subs = [0.5, 0.5, 0.12084000000000002, 0.12084000000000002, 0.20073333333333337, 0.20073333333333337, 0.30060000000000003, 0.30060000000000003, 0.3205733333333334, 0.3205733333333334, 0.3405466666666667, 0.3405466666666667, 0.36052000000000006, 0.36052000000000006, 0.36052000000000006, 0.3804933333333334, 0.3804933333333334, 0.3804933333333334, 0.40046666666666675, 0.40046666666666675, 0.40046666666666675, 0.42044000000000004, 0.42044000000000004, 0.42044000000000004, 0.4404133333333334, 0.4404133333333334, 0.4404133333333334, 0.4603866666666667, 0.4603866666666667, 0.4603866666666667, 0.4603866666666667, 0.4603866666666667, 0.48036000000000006, 0.48036000000000006, 0.48036000000000006, 0.48036000000000006, 0.48036000000000006, 0.5003333333333334, 0.5003333333333334, 0.5003333333333334, 0.5203066666666667, 0.5203066666666667, 0.5402800000000001, 0.5402800000000001, 0.5402800000000001, 0.5402800000000001, 0.5802266666666668, 0.5802266666666668, 0.6002000000000001, 0.6002000000000001, 0.6002000000000001, 0.6201733333333335, 0.6201733333333335, 0.6401466666666668, 0.6401466666666668, 0.6401466666666668, 0.66012, 0.66012, 0.7000666666666667, 0.7000666666666667, 0.7200400000000001, 0.7200400000000001, 0.7400133333333334, 0.7400133333333334, 0.7400133333333334, 0.8199066666666668, 0.8199066666666668, 0.8598533333333335, 0.8598533333333335, 0.9597200000000001, 0.9597200000000001, 1.13948, 1.13948, 0.5, 0.5, 0.5, 0.5, 0.5]

    lens_model_list_subs = ['SPEMD', 'SHEAR', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'CONVERGENCE', 'TNFW', 'TNFW', 'TNFW', 'TNFW', 'TNFW']
    kwargs_lens_subs = [{'theta_E': 1, 'center_x': 0, 'center_y': 0, 'e1': 0.03585665252864011, 'gamma': 2, 'e2': 0.08890716633399057}, {'e1': 0.010418890660015825, 'e2': -0.05908846518073248}, {'theta_Rs': 0.0012203790177275286, 'center_y': -1.7921266792794754, 'center_x': -0.467428772464865, 'r_trunc': 2.5100742846806376, 'Rs': 0.42756921312587975}, {'kappa_ext': -0.00021853515661623428}, {'theta_Rs': 0.003444595540289045, 'center_y': -1.1660392566119568, 'center_x': -0.6070357868139882, 'r_trunc': 2.417835479528742, 'Rs': 0.27390034146111647}, {'kappa_ext': -0.0004972032933313636}, {'theta_Rs': 0.00017962867638145414, 'center_y': 2.145909980330011, 'center_x': -1.3183957419851695, 'r_trunc': 0.4245824265642641, 'Rs': 0.08579863419499399}, {'kappa_ext': -5.809738518771412e-06}, {'theta_Rs': 0.007949754250755801, 'center_y': 0.3760585426093998, 'center_x': -0.811883175619956, 'r_trunc': 2.485243093325807, 'Rs': 0.3240052867417912}, {'kappa_ext': -0.001258248276369867}, {'theta_Rs': 0.0008560648040583813, 'center_y': 1.116689002839511, 'center_x': -0.171054835855971, 'r_trunc': 0.8063224513382014, 'Rs': 0.13123634786332194}, {'kappa_ext': -4.8345934002938364e-05}, {'theta_Rs': 0.002757377204910159, 'center_y': 1.0316360574746055, 'center_x': 0.5479061372953873, 'r_trunc': 1.3540498141093067, 'Rs': 0.19745468599952312}, {'theta_Rs': 0.0007601023920148339, 'center_y': -1.1896168559049465, 'center_x': -2.0447415408752847, 'r_trunc': 0.7440834527289049, 'Rs': 0.14391896114085437}, {'kappa_ext': -0.0002921367820221159}, {'theta_Rs': 0.0012374600242477613, 'center_y': -1.2661535072095242, 'center_x': -0.3482234175533696, 'r_trunc': 0.8752628623663938, 'Rs': 0.13386341473298335}, {'theta_Rs': 0.00025762095180453456, 'center_y': 1.7602724970899797, 'center_x': 0.28423581696323874, 'r_trunc': 0.41445590898392903, 'Rs': 0.08012569059917486}, {'kappa_ext': -8.193066358996805e-05}, {'theta_Rs': 0.00036950312854799644, 'center_y': -2.0953173757612276, 'center_x': -0.9783295029449985, 'r_trunc': 0.4874668122759245, 'Rs': 0.11252333630978899}, {'theta_Rs': 0.00097531347442386, 'center_y': -0.5994276391320326, 'center_x': -1.9545627350061117, 'r_trunc': 0.746091105035617, 'Rs': 0.11440136910541679}, {'kappa_ext': -6.41053410572328e-05}, {'theta_Rs': 0.003009363568540769, 'center_y': -2.783729931003233, 'center_x': 0.052687755114580945, 'r_trunc': 1.2956466116841792, 'Rs': 0.23436041899858734}, {'theta_Rs': 0.0009478537153768127, 'center_y': 0.7059618949951327, 'center_x': 1.0043864752455225, 'r_trunc': 0.7202429331189855, 'Rs': 0.12254536592196298}, {'kappa_ext': -0.0003335119151750039}, {'theta_Rs': 0.0017841588161385913, 'center_y': -0.48338123081136175, 'center_x': -0.16061650833148833, 'r_trunc': 0.9868803498941865, 'Rs': 0.21355404260934813}, {'theta_Rs': 0.00016089113868436881, 'center_y': 1.777036295869062, 'center_x': -2.2286205021674026, 'r_trunc': 0.2975628602007668, 'Rs': 0.06653640071102164}, {'kappa_ext': -0.00014125927042599306}, {'theta_Rs': 0.0002819078875813902, 'center_y': 0.03939203076600291, 'center_x': 0.3112617303065493, 'r_trunc': 0.3789181705647839, 'Rs': 0.08119737497712705}, {'theta_Rs': 0.00016688950334582618, 'center_y': 0.34178224699272636, 'center_x': -1.0004954897717653, 'r_trunc': 0.2899247051689978, 'Rs': 0.059536572514964005}, {'theta_Rs': 0.0013136505535720085, 'center_y': -0.833793364305014, 'center_x': 2.829086376669196, 'r_trunc': 0.812039385927823, 'Rs': 0.16467813540633575}, {'theta_Rs': 0.0006601545690250734, 'center_y': 0.7809224711319476, 'center_x': -1.3914637525409537, 'r_trunc': 0.579509805329494, 'Rs': 0.12361643321013822}, {'kappa_ext': -0.00012310344451643132}, {'theta_Rs': 0.00014796982284996508, 'center_y': 0.9854113305510049, 'center_x': 0.2346702138161839, 'r_trunc': 0.27194972356774993, 'Rs': 0.07064560928217409}, {'theta_Rs': 0.0007970214184985728, 'center_y': 2.217921734830792, 'center_x': 0.1035682732486876, 'r_trunc': 0.5846513362557996, 'Rs': 0.08755512805529835}, {'theta_Rs': 0.001110182413616029, 'center_y': 1.57728322965056, 'center_x': 2.060906834835671, 'r_trunc': 0.7048697110707712, 'Rs': 0.11980384412573769}, {'theta_Rs': 0.0011022296058103215, 'center_y': 2.794493117552481, 'center_x': -0.7420401445395466, 'r_trunc': 0.6780736149230973, 'Rs': 0.09406041777496249}, {'kappa_ext': -0.00013956772987649716}, {'theta_Rs': 0.00039853088111965084, 'center_y': -1.9044140075988603, 'center_x': 1.0868057199269683, 'r_trunc': 0.4274232489185921, 'Rs': 0.0976054217044671}, {'theta_Rs': 0.0012385201395769276, 'center_y': -0.2963758106627524, 'center_x': -1.9535424705937297, 'r_trunc': 0.7230440718491553, 'Rs': 0.12288384095401557}, {'kappa_ext': -7.737981944605546e-05}, {'theta_Rs': 0.000123886134078492, 'center_y': -0.35258209409142743, 'center_x': -2.139902847820809, 'r_trunc': 0.23588118123286977, 'Rs': 0.06328181960660267}, {'kappa_ext': -2.450307484761059e-06}, {'theta_Rs': 0.0028454887015175014, 'center_y': -1.3741872006488698, 'center_x': -1.420336453517475, 'r_trunc': 1.064766452905526, 'Rs': 0.21260420706209682}, {'theta_Rs': 0.00021809814951886356, 'center_y': 0.9859611891759236, 'center_x': 1.1682327954890823, 'r_trunc': 0.2970854059010245, 'Rs': 0.06287187293597275}, {'theta_Rs': 0.0027873255168534834, 'center_y': 0.6297316427438989, 'center_x': 0.7277786707814898, 'r_trunc': 0.9949319700413086, 'Rs': 0.13926447235945508}, {'kappa_ext': -0.0004170497032360246}, {'theta_Rs': 0.00015804937265727174, 'center_y': -1.8862676168532113, 'center_x': 0.9203054698110426, 'r_trunc': 0.24183250606912704, 'Rs': 0.05175465344943231}, {'kappa_ext': -2.972600709628487e-06}, {'theta_Rs': 0.0014117475835300028, 'center_y': 0.13974084246192642, 'center_x': 1.167736612232362, 'r_trunc': 0.662776288649373, 'Rs': 0.09322155090591085}, {'theta_Rs': 0.0007634363548092544, 'center_y': -0.7651738920084389, 'center_x': 1.4518718782968751, 'r_trunc': 0.5197621276110419, 'Rs': 0.10992171027751707}, {'kappa_ext': -9.235069992570643e-05}, {'theta_Rs': 0.0008392593392263917, 'center_y': 2.0031843762810317, 'center_x': 0.6331741701928624, 'r_trunc': 0.5209031416082539, 'Rs': 0.09193444057314737}, {'kappa_ext': -3.163033144841063e-05}, {'theta_Rs': 0.0006971945854860033, 'center_y': 0.0816236424856843, 'center_x': -0.5254097404188737, 'r_trunc': 0.46886624635534657, 'Rs': 0.08578069120822493}, {'theta_Rs': 0.0007972653250693336, 'center_y': -0.33303005406405123, 'center_x': -1.1036894260160657, 'r_trunc': 0.5173817495749006, 'Rs': 0.11975438865210465}, {'kappa_ext': -5.6953269127511145e-05}, {'theta_Rs': 0.00011654472264938334, 'center_y': 0.2534622906920973, 'center_x': -0.6656913018607555, 'r_trunc': 0.19905965851116747, 'Rs': 0.05819630675622407}, {'kappa_ext': -1.9961375620040585e-06}, {'theta_Rs': 0.0005663088998894857, 'center_y': -0.29388460044874787, 'center_x': -0.763596139084235, 'r_trunc': 0.40959339173166437, 'Rs': 0.08331560739657465}, {'kappa_ext': -1.7712179908722264e-05}, {'theta_Rs': 0.002755858166487574, 'center_y': -0.9946920497009132, 'center_x': -0.6924499839667908, 'r_trunc': 0.8616633002382044, 'Rs': 0.13953060884872917}, {'kappa_ext': -0.00016597154576195204}, {'theta_Rs': 0.00028350956396907233, 'center_y': 0.3837755732142354, 'center_x': -0.958150920550796, 'r_trunc': 0.2919492405858599, 'Rs': 0.07870784687133421}, {'theta_Rs': 0.00038659240361814216, 'center_y': -0.6471039860804096, 'center_x': 1.2248123566493518, 'r_trunc': 0.3412148173189984, 'Rs': 0.09286237857477006}, {'kappa_ext': -1.8060542966084125e-05}, {'theta_Rs': 0.0010795947563896357, 'center_y': -0.28418789204026274, 'center_x': 0.3333146468853628, 'r_trunc': 0.5420533473628893, 'Rs': 0.1282971418392103}, {'kappa_ext': -4.7125524422044866e-05}, {'theta_Rs': 0.0005029698616180579, 'center_y': -0.5685143624948263, 'center_x': 0.32660273403421014, 'r_trunc': 0.36231717444426736, 'Rs': 0.07934878593382862}, {'kappa_ext': -1.4289005834198816e-05}, {'theta_Rs': 0.0018134964980950031, 'center_y': 0.03688026445365063, 'center_x': -0.3787060768191913, 'r_trunc': 0.6697610245609084, 'Rs': 0.12970506560657954}, {'kappa_ext': -9.107318360930914e-05}, {'theta_Rs': 0.0008284224675439594, 'center_y': -0.35220847719747217, 'center_x': -0.2515273195529677, 'r_trunc': 0.4875848940650852, 'Rs': 0.10602274184281299}, {'kappa_ext': -3.159228806040084e-05}, {'theta_Rs': 0.000465088441538369, 'center_y': -0.16502502764441757, 'center_x': 0.1714776762150703, 'r_trunc': 0.2219693133232066, 'Rs': 0.10678779854183958}, {'theta_Rs': 0.0016556410730280769, 'center_y': -1.001169835842999, 'center_x': 0.2613444295318975, 'r_trunc': 0.22844476249560336, 'Rs': 0.14427899554679524}, {'theta_Rs': 0.0003829431617571525, 'center_y': -0.7467699211563197, 'center_x': -0.5764483343720752, 'r_trunc': 0.22575378232427776, 'Rs': 0.06727699653224929}, {'theta_Rs': 0.0018514932315065328, 'center_y': -0.9148075521072269, 'center_x': -2.696921580033914, 'r_trunc': 0.7338222007356427, 'Rs': 0.19160824243542735}, {'theta_Rs': 0.005345187143413565, 'center_y': 0.42835609495699994, 'center_x': -1.5304433660274317, 'r_trunc': 0.3478854843826851, 'Rs': 0.23268032576843709}]

    optimizer_simple = Optimizer(x_pos, y_pos, magnification_target=magnification, redshift_list=redshift_list_simple,
                                 lens_model_list=lens_model_list_simple, kwargs_lens=kwargs_lens_simple, multiplane=True, verbose=True,
                                 z_source=1.5,z_main=0.5,astropy_instance=cosmo)

    optimizer_subs = Optimizer(x_pos, y_pos, magnification_target=magnification, redshift_list=redshift_list_subs,
                               lens_model_list=lens_model_list_subs, kwargs_lens=kwargs_lens_subs, multiplane=True, verbose=True,
                               z_source=1.5,z_main=0.5,astropy_instance=cosmo)

    def test_multi_plane_incorrect_model(self,tol=0.003):
        """
        The data was created with a full LOS realization, so the simple lens model is actually incorrect
        :param tol: image position tolerance
        :return:
        """
        kwargs_lens, source, [x_image,y_image] = self.optimizer_simple.optimize(n_particles=50,n_iterations=200)
        index = sort_image_index(x_image, y_image, self.x_pos, self.y_pos)

        x_image = x_image[index]
        y_image = y_image[index]

        dx = np.absolute(x_image - self.x_pos)
        dy = np.absolute(y_image - self.y_pos)

        # magnifications will be super-wrong
        npt.assert_array_less(dx, [tol] * len(dx))
        npt.assert_array_less(dy, [tol] * len(dy))

    def test_multi_plane_correct_model(self):
        """
        Should be a near perfect fit since the LOS model is the same used to create the data.
        :return:
        """
        kwargs_lens, source, [x_image,y_image] = self.optimizer_subs.optimize(n_particles=50,n_iterations=300)

        mags = self.optimizer_subs.optimizer_amoeba.lensModel.magnification(x_image, y_image, kwargs_lens)
        mags = np.absolute(mags)
        mags *= max(mags) ** -1

        index = sort_image_index(x_image, y_image, self.x_pos, self.y_pos)
        x_image = x_image[index]
        y_image = y_image[index]
        mags = mags[index]

        # everything should match
        npt.assert_almost_equal(x_image, self.x_pos, decimal=3)
        npt.assert_almost_equal(y_image, self.y_pos, decimal=3)
        npt.assert_almost_equal(self.magnification, mags, decimal=2)

if __name__ == '__main__':
    pytest.main()

