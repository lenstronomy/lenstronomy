__author__ = 'sibirrer'

# this file contains a class to manage the finder que.
# The job of this class is to select a finder que and to return it in a standardised format

import copy
from lenstronomy.FunctionSet.dipole import Dipole_util

class FinderQue(object):

    def __init__(self, position_finder, kwargs_finder):
        self.position_finder = position_finder
        self.num_cpu = kwargs_finder['num_cpu']
        self.mpi_monch = kwargs_finder.get('mpi_monch', False)
        [self.param_list_catalogue, self.param_list_lens_light_init, self.param_list_image, self.param_list_WLS, self.param_list_lens_light, self.param_list_perturb, self.param_list_shapelets, self.param_list_all, self.param_list_all_2, self.param_list_all_3, self.param_list_shear, self.param_list_beta, self.param_list_final, self.param_list_mcmc, self.param_list_detect, self.param_list_psf_iteration] = [None]*16
        [self.chain_catalogue, self.chain_lens_light_init, self.chain_image, self.chain_WLS, self.chain_lens_light, self.chain_perturb, self.chain_shapelets, self.chain_all, self.chain_all_2, self.chain_all_3, self.chain_shear, self.chain_beta, self.chain_final, self.chain_detect, self.chain_psf_iteration, self.samples] = [None]*16
        self.dipole_util = Dipole_util()

    def finder_que(self, kwargs_finder):
        que_name = kwargs_finder.get('finder_que', 'DEFAULT')
        if que_name == 'SUBCLUMP_DETECT_ONLY':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.position_finder.return_init()
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_queue_detect(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        elif que_name == 'FINAL':
            n_walkers = kwargs_finder['final_walk']
            n_iter = kwargs_finder['final_iter']
            lens_result, source_result, psf_result, lens_light_result, else_result = self.position_finder.return_init()
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final, self.param_list_final = self.position_finder.find_param_final(
                lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu,
                mpi_monch=self.mpi_monch, dipole=False)

        else:
            if kwargs_finder.get('catalogue_iter', True):
                n_iter = kwargs_finder['cat_iter']
                n_walkers = kwargs_finder['cat_walk']
                lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_catalogue, self.param_list_catalogue = self.position_finder.find_param_catalogue(n_walkers, n_iter, mpi_monch=False)
            else:
                lens_result, source_result, psf_result, lens_light_result, else_result = self.position_finder.return_init()

        if que_name == 'CATALOGUE':
            pass
        elif que_name == 'DEFAULT':
            # sample only SPEP profile
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_default(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'ARC':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_arc(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'ARC_FOREGROUND':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_arc_foreground(kwargs_finder,
                                                                                                         lens_result,
                                                                                                         source_result,
                                                                                                         psf_result,
                                                                                                         lens_light_result,
                                                                                                         else_result)
        elif que_name == 'ARC_POINT':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_queue_arc_point(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'SUBCLUMP_ONLY':
            # sample with subclump but without shapelets
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_subclump_only(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'SUBCLUMP_PERTURB':
            # sampling with subclump and 5 additional shapelets
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_subclump_perturb(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'SHAPELETS_LONG':
            # sampling in addition with multiple lens shapelets (MCMC possible)
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_shapelets_long(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'DIPOLE':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_dipole(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'SHAPELETS_ONLY':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_shapelets_only(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'SUYU':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_suyu(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'SHAPELETS_ADDING':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_shapelets_adding(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        elif que_name == 'SUBCLUMP_DETECT':
            lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_queue_subclump(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)


        if kwargs_finder.get('psf_iteration', False) is True:
            num_order = kwargs_finder['shapelet_order_psf']
            n_iter = kwargs_finder['psf_iter']
            n_walkers = kwargs_finder['psf_walk']
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_psf_iteration, self.param_list_psf_iteration = self.position_finder.find_param_psf_iteration(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

            n_walkers = kwargs_finder['final_walk']
            n_iter = kwargs_finder['final_iter']
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final,  self.param_list_final = self.position_finder.find_param_final(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, dipole=False)

        if kwargs_finder.get('source_beta_perturb', False) is True:
            n_iter = kwargs_finder['source_beta_iter']
            n_walkers = kwargs_finder['source_beta_walk']
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_beta, self.param_list_beta = self.position_finder.find_param_beta_source(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        param_list = [self.param_list_catalogue, self.param_list_lens_light_init, self.param_list_image, self.param_list_WLS, self.param_list_lens_light, self.param_list_perturb, self.param_list_shapelets, self.param_list_all, self.param_list_all_2, self.param_list_all_3, self.param_list_shear, self.param_list_final, self.param_list_detect, self.param_list_mcmc, self.param_list_psf_iteration]
        chain_list = [self.chain_catalogue, self.chain_lens_light_init, self.chain_image, self.chain_WLS, self.chain_lens_light, self.chain_perturb, self.chain_shapelets, self.chain_all, self.chain_all_2, self.chain_all_3, self.chain_shear, self.chain_final, self.chain_detect, self.chain_psf_iteration]
        return lens_result, source_result, psf_result, lens_light_result, else_result, chain_list, param_list, self.samples

    def finder_lens_light(self, n_iter, n_walkers):
        lens_light_result, self.chain_lens_light_init, self.param_list_lens_light_init = self.position_finder.find_param_lens_light_init(
            n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)
        return lens_light_result, self.chain_lens_light_init, self.param_list_lens_light_init

    def finder_que_default(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        n_iter = kwargs_finder['lens_light_init_iter']
        n_walkers = kwargs_finder['lens_light_init_walk']
        lens_light_result, self.chain_lens_light_init, self.param_list_lens_light_init = self.position_finder.find_param_lens_light_init(n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        if kwargs_finder.get('lens_sersic', False) is True:
            n_iter = kwargs_finder['image_iter']
            n_walkers = kwargs_finder['image_walk']
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_image, self.param_list_image = self.position_finder.find_param_image(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        num_order = kwargs_finder['shapelet_order_wls']
        n_iter = kwargs_finder['wls_iter']
        n_walkers = kwargs_finder['wls_walk']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_WLS, self.param_list_WLS = self.position_finder.find_param_WLS(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        num_order = kwargs_finder['shapelet_order_wls']
        n_iter = kwargs_finder['lens_light_iter']
        n_walkers = kwargs_finder['lens_light_walk']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_lens_light, self.param_list_lens_light = self.position_finder.find_param_lens_light_combined(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_arc(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        n_iter = kwargs_finder['lens_light_init_iter']
        n_walkers = kwargs_finder['lens_light_init_walk']
        lens_light_result, self.chain_lens_light_init, self.param_list_lens_light_init = self.position_finder.find_param_lens_light_init(n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        num_order = kwargs_finder['shapelet_order_wls']
        n_iter = kwargs_finder['wls_iter']
        n_walkers = kwargs_finder['wls_walk']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_WLS, self.param_list_WLS = self.position_finder.find_param_arc(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        num_order = kwargs_finder['shapelet_order_wls']
        n_iter = kwargs_finder['lens_light_iter']
        n_walkers = kwargs_finder['lens_light_walk']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_lens_light, self.param_list_lens_light = self.position_finder.find_param_lens_light_combined(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        n_walkers = kwargs_finder['all_walk']
        n_iter = kwargs_finder['all_iter']
        num_order = kwargs_finder['shapelet_order_all']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all, self.param_list_all= self.position_finder.find_param_all_arc(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)
        #self.position_finder.kwargs_options["foreground_shear"] = True
        #n_walkers = kwargs_finder['all_walk']
        #n_iter = kwargs_finder['all_iter']
        #num_order = kwargs_finder['shapelet_order_all']
        #lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all, self.param_list_all= self.position_finder.find_param_all_arc(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_arc_foreground(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_arc(kwargs_finder,
                                                                                                         lens_result,
                                                                                                         source_result,
                                                                                                         psf_result,
                                                                                                         lens_light_result,
                                                                                                         else_result)
        self.position_finder.kwargs_options["foreground_shear"] = True
        n_walkers = kwargs_finder['foreground_walk']
        n_iter = kwargs_finder['foreground_iter']
        num_order = kwargs_finder['shapelet_order_all']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all, self.param_list_all = self.position_finder.find_param_all_arc(
            lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order,
            numThreads=self.num_cpu, mpi_monch=self.mpi_monch)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_queue_arc_point(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_default(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        n_walkers = kwargs_finder['all_walk']
        n_iter = kwargs_finder['all_iter']
        num_order = kwargs_finder['shapelet_order_all']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_WLS, self.param_list_WLS = self.position_finder.find_param_all_arc_point(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_subclump_only(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        """
        workflow of finding the position in parameter space
        :return: position in parameter space
        """

        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_default(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        if kwargs_finder['Lens_perturb'] == 'NFW' or kwargs_finder['Lens_perturb'] == 'SIS' or kwargs_finder['Lens_perturb'] == 'SPP' or kwargs_finder['Lens_perturb'] == 'SPP_SHAPELETS':
            n_walkers = kwargs_finder['perturb_walk']
            n_iter = kwargs_finder['perturb_iter']
            num_order = kwargs_finder['shapelet_order_perturb']
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_perturb,  self.param_list_perturb = self.position_finder.find_param_perturb(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump=kwargs_finder['Lens_perturb'])
        else:
            raise ValueError('%s is not a valid subclump!' %(kwargs_finder['Lens_perturb']))
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_subclump_perturb(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_subclump_only(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        num_order = kwargs_finder['shapelet_order_perturb']
        n_walkers = kwargs_finder['shapelet_lens_walk']
        n_iter = kwargs_finder['shapelet_lens_iter']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_shapelets, self.param_list_shapelets = self.position_finder.find_param_lens_pertrub_shapelets(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump=kwargs_finder['Lens_perturb'])

        n_walkers = kwargs_finder['all_walk']
        n_iter = kwargs_finder['all_iter']
        num_order = kwargs_finder['shapelet_order_all']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all,  self.param_list_all = self.position_finder.find_param_all(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_shapelets_long(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        """
        workflow of finding the position in parameter space
        :return: position in parameter space
        """
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_subclump_perturb(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        n_walkers = kwargs_finder['shapelet_perturb_walk_1']
        n_iter = kwargs_finder['shapelet_perturb_iter_1']
        num_order = kwargs_finder['shapelet_order_add_1']
        num_shapelets_lens = kwargs_finder['num_shapelets_lens_1']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all_2,  self.param_list_all_2 = self.position_finder.find_param_add_shapelets(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, num_shapelets_lens, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump=kwargs_finder['Lens_perturb'])

        n_walkers = kwargs_finder['shapelet_perturb_walk_2']
        n_iter = kwargs_finder['shapelet_perturb_iter_2']
        num_order = kwargs_finder['shapelet_order_add_2']
        num_shapelets_lens = kwargs_finder['num_shapelets_lens_2']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all_3,  self.param_list_all_3 = self.position_finder.find_param_add_shapelets(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, num_shapelets_lens, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump=kwargs_finder['Lens_perturb'])

        n_walkers = kwargs_finder['shear_walk']
        n_iter = kwargs_finder['shear_iter']
        num_order = kwargs_finder['shapelet_order_shear']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_shear,  self.param_list_shear = self.position_finder.find_param_external_shear(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        n_walkers = kwargs_finder['final_walk']
        n_iter = kwargs_finder['final_iter']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final,  self.param_list_final = self.position_finder.find_param_final(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, dipole=False)

        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_dipole(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        """
        finder que for dipole component between clumps
        :return:
        """
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_default(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        n_walkers = kwargs_finder['coupling_walk']
        n_iter = kwargs_finder['coupling_iter']
        num_order = kwargs_finder['shapelet_order_coupling']

        lens_result['center_x_spp'] = lens_light_result['center_x_2'].copy()
        lens_result['center_y_spp'] = lens_light_result['center_y_2'].copy()
        lens_result['phi_E_spp'] = self.position_finder.kwargs_lens_clump_init['phi_E_spp']
        lens_result['gamma_spp'] = self.position_finder.kwargs_lens_clump_init['gamma_spp']
        lens_result['phi_dipole'] = self.dipole_util.angle(lens_result['center_x'], lens_result['center_y'], lens_result['center_x_spp'], lens_result['center_y_spp'])

        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_shapelets, self.param_list_shapelets = self.position_finder.find_param_lens_pertrub_shapelets(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump=kwargs_finder['Lens_perturb'], dipole=True)

        n_walkers = kwargs_finder['all_walk']
        n_iter = kwargs_finder['all_iter']
        num_order = kwargs_finder['shapelet_order_all']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all,  self.param_list_all = self.position_finder.find_param_all(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        n_walkers = kwargs_finder['shear_walk']
        n_iter = kwargs_finder['shear_iter']
        num_order = kwargs_finder['shapelet_order_shear']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_shear,  self.param_list_shear = self.position_finder.find_param_external_shear(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        n_walkers = kwargs_finder['final_walk']
        n_iter = kwargs_finder['final_iter']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final,  self.param_list_final = self.position_finder.find_param_final(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, dipole=True)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_shapelets_only(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_default(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        n_walkers = kwargs_finder['shapelet_lens_walk']
        n_iter = kwargs_finder['shapelet_lens_iter']
        num_order = kwargs_finder['shapelet_order_perturb']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_shapelets, self.param_list_shapelets = self.position_finder.find_param_lens_pertrub_shapelets(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump='NONE')

        n_walkers = kwargs_finder['all_walk']
        n_iter = kwargs_finder['all_iter']
        num_order = kwargs_finder['shapelet_order_all']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all,  self.param_list_all = self.position_finder.find_param_all(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump='NONE', shapelet_lens=True)

        n_walkers = kwargs_finder['shapelet_perturb_walk_1']
        n_iter = kwargs_finder['shapelet_perturb_iter_1']
        num_order = kwargs_finder['shapelet_order_add_1']
        num_shapelets_lens = kwargs_finder['num_shapelets_lens_1']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all_2,  self.param_list_all_2 = self.position_finder.find_param_add_shapelets(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, num_shapelets_lens, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump='NONE')

        n_walkers = kwargs_finder['shapelet_perturb_walk_2']
        n_iter = kwargs_finder['shapelet_perturb_iter_2']
        num_order = kwargs_finder['shapelet_order_add_2']
        num_shapelets_lens = kwargs_finder['num_shapelets_lens_2']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_all_3,  self.param_list_all_3 = self.position_finder.find_param_add_shapelets(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, num_shapelets_lens, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, clump='NONE')

        n_walkers = kwargs_finder['final_walk']
        n_iter = kwargs_finder['final_iter']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final,  self.param_list_final = self.position_finder.find_param_final(lens_result, source_result, psf_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, dipole=False)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_suyu(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_default(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        n_walkers = kwargs_finder['perturb_walk']
        n_iter = kwargs_finder['perturb_iter']
        num_order = kwargs_finder['shapelet_order_perturb']
        num_shapelet_lens = 0
        beta_lens = kwargs_finder['shapelet_lens_beta']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_perturb, self.param_list_perturb = self.position_finder.find_param_suyu_add(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, num_shapelet_lens, beta_lens, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        n_walkers = kwargs_finder['final_walk']
        n_iter = kwargs_finder['final_iter']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final, self.param_list_final = self.position_finder.find_param_final(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, dipole=False)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_que_shapelets_adding(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_suyu(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)

        n_walkers = kwargs_finder['perturb_walk']
        n_iter = kwargs_finder['perturb_iter']
        num_order = kwargs_finder['shapelet_order_perturb']
        num_shapelet_lens = kwargs_finder['num_shapelets_lens']
        if num_shapelet_lens <= 0:
            return lens_result, source_result, psf_result, lens_light_result, else_result
        else:
            beta_lens = kwargs_finder['shapelet_lens_beta']
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_perturb, self.param_list_perturb = self.position_finder.find_param_suyu_add(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, num_order, num_shapelet_lens, beta_lens, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

            n_walkers = kwargs_finder['final_walk']
            n_iter = kwargs_finder['final_iter']
            lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final,  self.param_list_final = self.position_finder.find_param_final(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, dipole=False)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_queue_dipole_adding(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_suyu(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        #TODO add dipole perturbation (probably only with shapelets perturb involved)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_queue_subclump(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        lens_result, source_result, psf_result, lens_light_result, else_result = self.finder_que_shapelets_adding(kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result)
        n_walkers = kwargs_finder['subclump_walk']
        n_iter = kwargs_finder['subclump_iter']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_detect, self.param_list_detect = self.position_finder.find_subclump(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)
        return lens_result, source_result, psf_result, lens_light_result, else_result

    def finder_queue_detect(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        n_walkers = kwargs_finder['subclump_walk']
        n_iter = kwargs_finder['subclump_iter']
        else_result_new = else_result.copy()
        del else_result_new['phi_E_clump']
        del else_result_new['x_clump']
        del else_result_new['y_clump']
        del else_result_new['r_trunc']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_detect, self.param_list_detect = self.position_finder.find_subclump(lens_result, source_result, lens_light_result, else_result_new, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        n_walkers = kwargs_finder['final_walk']
        n_iter = kwargs_finder['final_iter']
        lens_result, source_result, psf_result, lens_light_result, else_result, self.chain_final,  self.param_list_final = self.position_finder.find_param_final(lens_result, source_result, lens_light_result, else_result, n_walkers, n_iter, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, dipole=False)

        return lens_result, source_result, psf_result, lens_light_result, else_result