from lenstronomy.Workflow.multi_band_manager import MultiBandUpdateManager
from lenstronomy.Workflow.update_manager import UpdateManager
import pytest


class TestUpdateManager(object):

    def setup_method(self):
        kwargs_model = {'lens_model_list': ['SHEAR', 'SHEAR'], 'source_light_model_list': ['UNIFORM'],
                        'lens_light_model_list': ['UNIFORM'],
                        'optical_depth_model_list': []}
        kwargs_constraints ={}
        kwargs_likelihood = {}
        kwargs_params = {}
        lens_init = [{'e1': 0, 'e2': 0}, {'e1': 0, 'e2': 0}]
        lens_sigma = [{'e1': 0.1, 'e2': 0.1}, {'e1': 0.1, 'e2': 0.1}]
        lens_fixed = [{'ra_0': 0, 'dec_0': 0}, {'ra_0': 0, 'dec_0': 0}]
        lens_lower = [{'e1': -1, 'e2': -1}, {'e1': -1, 'e2': -1}]
        lens_upper = [{'e1': 1, 'e2': 1}, {'e1': 1, 'e2': 1}]
        kwargs_params['lens_model'] = [lens_init, lens_sigma, lens_fixed, lens_lower, lens_upper]
        kwargs_params['source_model'] = [[{}], [{}], [{}], [{}], [{}]]
        kwargs_params['lens_light_model'] = [[{}], [{}], [{}], [{}], [{}]]
        kwargs_params['special'] = [{'special1': 1}, {'special1': 1}, {'special1': 0.1}, {'special1': 0}, {'special1': 1}]
        kwargs_params['extinction_model'] = [[], [], [], [], []]
        self.manager = UpdateManager(kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

    def test_none_mamager(self):
        manager = MultiBandUpdateManager(kwargs_model={}, kwargs_constraints={}, kwargs_likelihood={}, kwargs_params={}, num_bands=0)
        results = manager.best_fit()
        assert len(results['kwargs_lens']) == 0

    def test_init_kwargs(self):
        kwargs_init = self.manager.init_kwargs
        assert kwargs_init['kwargs_lens'][0]['e1'] == 0

    def test_sigma_kwargs(self):
        kwargs_sigma = self.manager.sigma_kwargs
        assert kwargs_sigma['kwargs_lens'][0]['e1'] == 0.1

    def test_update_parameter_state(self):
        self.manager.update_param_state(kwargs_lens=[{'e1': -2, 'e2': 0}, {'e1': 2, 'e2': 0}])
        kwargs_temp = self.manager.parameter_state
        assert kwargs_temp['kwargs_lens'][0]['e1'] == -2
        self.manager.set_init_state()
        kwargs_temp = self.manager.parameter_state
        assert kwargs_temp['kwargs_lens'][0]['e1'] == 0

    def test_update_param_value(self):
        self.manager.update_param_value(lens=[[1, ['e1'], [0.029]]])
        kwargs_temp = self.manager.parameter_state
        assert kwargs_temp['kwargs_lens'][1]['e1'] == 0.029

    def test_param_class(self):
        param_class = self.manager.param_class
        num_param, param_names = param_class.num_param()
        assert num_param == 4

    def test_best_fit(self):
        kwargs_result = self.manager.best_fit(bijective=True)
        assert kwargs_result['kwargs_lens'][0]['e1'] == 0

    def test_update_options(self):
        self.manager.update_options(kwargs_model=None, kwargs_constraints={'test': 'test'}, kwargs_likelihood=None)
        assert self.manager.kwargs_constraints['test'] == 'test'

        self.manager.update_options(kwargs_model={'test': 'test'}, kwargs_constraints=None, kwargs_likelihood=None)
        assert self.manager.kwargs_model['test'] == 'test'

    def test_update_limits(self):
        self.manager.update_limits(change_source_lower_limit=[[0, ['test'], [-1]]], change_source_upper_limit=[[0, ['test'], [1]]])
        self.manager.update_limits(change_lens_lower_limit=[[0, ['e1'], [-0.9]]], change_lens_upper_limit=[[0, ['e1'], [0.9]]])
        upper_lens, upper_source, _, _, _, _ = self.manager._upper_kwargs
        assert upper_source[0]['test'] == 1
        assert upper_lens[0]['e1'] == 0.9

    def test_update_sigmas(self):
        self.manager.update_sigmas(change_sigma_source=[[0, ['test'], [1]]],
                                   change_sigma_lens=[[0, ['test'], [2]]])
        self.manager.update_sigmas(change_sigma_lens_light=[[0, ['e1'], [-0.9]]],
                                   change_sigma_lens=[[0, ['e1'], [0.9]]])
        upper_lens, upper_source, _, _, _, _ = self.manager._upper_kwargs
        assert self.manager._lens_sigma[0]['test'] == 2
        assert self.manager._lens_sigma[0]['e1'] == 0.9

    def test_update_fixed(self):
        lens_add_fixed = [[0, ['e1'], [-1]]]
        self.manager.update_fixed(lens_add_fixed=lens_add_fixed)
        assert self.manager._lens_fixed[0]['e1'] == -1

        lens_add_fixed = [[0, ['e2']]]
        self.manager.update_fixed(lens_add_fixed=lens_add_fixed)
        assert self.manager._lens_fixed[0]['e2'] == 0

        lens_remove_fixed = [[0, ['e1']]]
        self.manager.update_fixed(lens_remove_fixed=lens_remove_fixed)
        assert 'e1' not in self.manager._lens_fixed[0]

        assert 'special1' in self.manager._special_fixed
        self.manager.update_fixed(special_remove_fixed=['special1'])
        assert 'special1' not in self.manager._special_fixed

        self.manager.update_fixed(special_add_fixed=['special1'])
        assert self.manager._special_fixed['special1'] == 1

        self.manager.update_fixed(special_add_fixed=['special1'])
        assert self.manager._special_fixed['special1'] == 1

    def test_update_logsampling(self):
        self.manager.update_options(kwargs_model={}, kwargs_constraints={'log_sampling_lens': [[0, ['e1']]]}, kwargs_likelihood = {})
        assert self.manager.param_class.lensParams.kwargs_logsampling[0] == ['e1']

    def test_fix_image_parameters(self):
        self.manager.fix_image_parameters(image_index=0)
        assert 1 == 1


if __name__ == '__main__':
    pytest.main()
