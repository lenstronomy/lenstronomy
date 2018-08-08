import numpy as np

class FixedPowerLaw_Shear(object):

    def __init__(self,lens_model_list,kwargs_lens,xpos,ypos):

        assert lens_model_list[0] in ['SPEMD','SPEP']
        assert lens_model_list[1] == 'SHEAR'
        self.Ntovary = 2
        self.k_start = 2
        self.tovary_indicies = [0,1]
        self.kwargs_lens = kwargs_lens

        self.theta_E_start = self._estimate_theta_E(xpos,ypos)

        self.param_names = [['theta_E', 'center_x', 'center_y', 'e1', 'e2','gamma'], ['e1', 'e2']]
        self.fixed_names = [['gamma'], []]
        self.fixed_values = [{'gamma': kwargs_lens[0]['gamma']}, {}]

    def _estimate_theta_E(self,ximg,yimg):

        dis = []
        xinds,yinds = [0,0,0,1,1,2],[1,2,3,2,3,3]

        for (i,j) in zip(xinds,yinds):

            dx,dy = ximg[i] - ximg[j], yimg[i] - yimg[j]
            dr = (dx**2+dy**2)**0.5
            dis.append(dr)
        dis = np.array(dis)

        greatest = np.argmax(dis)
        dr_greatest = dis[greatest]
        dis[greatest] = 0

        second_greatest = np.argmax(dis)
        dr_second = dis[second_greatest]

        return 0.5*(dr_greatest*dr_second)**0.5

    def get_param_ranges(self,reoptimize=False):

        if reoptimize:

            e1_e2 = 0.01
            e1_e2_shear = 0.01
            theta_E = 0.005
            center = 0.005

            low_e1 = self.kwargs_lens[0]['e1'] - e1_e2
            low_e2 = self.kwargs_lens[0]['e2'] - e1_e2
            hi_e1 = self.kwargs_lens[0]['e1'] + e1_e2
            hi_e2 = self.kwargs_lens[0]['e2'] + e1_e2

            low_shear_e1 = self.kwargs_lens[1]['e1'] - e1_e2_shear
            high_shear_e1 = self.kwargs_lens[1]['e1'] + e1_e2_shear
            low_shear_e2 = self.kwargs_lens[1]['e2'] - e1_e2_shear
            high_shear_e2 = self.kwargs_lens[1]['e2'] + e1_e2_shear

            low_Rein = self.kwargs_lens[0]['theta_E'] - theta_E
            hi_Rein = self.kwargs_lens[0]['theta_E'] + theta_E

            low_centerx = self.kwargs_lens[0]['center_x'] - center
            hi_centerx = self.kwargs_lens[0]['center_x'] + center
            low_centery = self.kwargs_lens[0]['center_y'] - center
            hi_centery = self.kwargs_lens[0]['center_y'] + center

        else:

            low_e1 = -0.3
            low_e2 = low_e1
            hi_e1 = 0.3
            hi_e2 = hi_e1

            low_shear_e1 = -0.065
            high_shear_e1 = 0.065
            low_shear_e2 = low_shear_e1
            high_shear_e2 = high_shear_e1

            low_Rein = self.theta_E_start - 0.1
            hi_Rein = self.theta_E_start + 0.1

            low_centerx = -0.015
            hi_centerx = 0.015
            low_centery = low_centerx
            hi_centery = hi_centerx

        sie_list_low = [low_Rein, low_centerx, low_centery, low_e1, low_e2]
        sie_list_high = [hi_Rein, hi_centerx, hi_centery, hi_e1, hi_e2]
        shear_list_low = [low_shear_e1, low_shear_e2]
        shear_list_high = [high_shear_e1, high_shear_e2]

        return sie_list_low+shear_list_low,sie_list_high+shear_list_high





