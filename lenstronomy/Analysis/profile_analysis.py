
from lenstronomy.Util import class_creator
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Analysis.light_profile import LightProfileAnalysis


class ProfileAnalysis(object):
    """
    class that bundles the different profile analysis routines from the global model configurations
    """
    def __init__(self, kwargs_model):

        self.LensModel, self.SourceModel, self.LensLightModel, self.PointSource, extinction_class = class_creator.create_class_instances(all_models=True, **kwargs_model)
        self.kwargs_model = kwargs_model
        self.lensProfile = LensProfileAnalysis(lens_model=self.LensModel)
        self.lensLightProfile = LightProfileAnalysis(light_model=self.LensLightModel)
        self.sourceLightProfile = LightProfileAnalysis(light_model=self.SourceModel)
