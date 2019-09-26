from lenstronomy.Workflow.multi_band_manager import MultiBandUpdateManager
import pytest


class TestUpdateManager(object):

    def setup(self):
        kwargs_model = {'lens_model_list': ['SHEAR', 'SHEAR'], 'source_light_model_list': ['UNIFORM'], 'index_lens_model_list': [[0], [1]]}
        kwargs_constraints ={}
        kwargs_likelihood = {}
        kwargs_params = {}
        lens_init = [{'e1': 0, 'e2': 0, 'ra_0': 0, 'dec_0': 0}, {'e1': 0, 'e2': 0, 'ra_0': 0, 'dec_0': 0}]
        lens_sigma = [{'e1': 0.1, 'e2': 0.1}, {'e1': 0.1, 'e2': 0.1}]
        lens_fixed = [{'ra_0': 0, 'dec_0': 0}, {'ra_0': 0, 'dec_0': 0}]
        lens_lower = [{'e1': -1, 'e2': -1}, {'e1': -1, 'e2': -1}]
        lens_upper = [{'e1': 1, 'e2': 1}, {'e1': 1, 'e2': 1}]
        kwargs_params['lens_model'] = [lens_init, lens_sigma, lens_fixed, lens_lower, lens_upper]
        kwargs_params['source_model'] = [[{}], [{}], [{}], [{}], [{}]]
        kwargs_params['special'] = [{'special1': 1}, {'special1': 1}, {'special1': 0.1}, {'special1': 0}, {'special1': 1}]
        self.manager = MultiBandUpdateManager(kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, num_bands=2)

    def test_none_mamager(self):
        manager = MultiBandUpdateManager(kwargs_model={}, kwargs_constraints={}, kwargs_likelihood={}, kwargs_params={}, num_bands=0)
        results = manager.best_fit()
        assert len(results['kwargs_lens']) == 0

    def test_keep_frame_fixed(self):
        frame_list_fixed = [0]
        assert 'e1' not in self.manager._lens_fixed[0]
        self.manager.keep_frame_fixed(frame_list_fixed)
        assert 'e1' in self.manager._lens_fixed[0]

        self.manager.undo_frame_fixed(frame_list=[0])
        assert 'e1' not in self.manager._lens_fixed[0]
        assert 'ra_0' in self.manager._lens_fixed[0]

    def test_fix_not_computed(self):

        self.manager.fix_not_computed(free_bands=[False, True])
        print(self.manager._lens_fixed)
        assert 'e1' in self.manager._lens_fixed[0]
        assert 'ra_0' in self.manager._lens_fixed[0]
        assert 'e1' not in self.manager._lens_fixed[1]


if __name__ == '__main__':
    pytest.main()
