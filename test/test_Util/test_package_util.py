import types

def _remove_cached_lenstronomy():
    """Removes lenstronomy from the module cache,
    if it was already imported.
    This is needed to reset the short / laconic states.
    """
    import sys
    if 'lenstronomy' in sys.modules:
        del sys.modules['lenstronomy']


def test_short():
    _remove_cached_lenstronomy()
    import lenstronomy as ls

    # Without running .short(), nothing is changed:
    assert not hasattr(ls, 'LensModel')

    ls.short()

    # We can access submodules as symbols...
    assert hasattr(ls, 'LensModel')
    assert isinstance(ls.LensModel, types.ModuleType)

    # ... also recursively.
    assert hasattr(ls.LensModel, 'lens_model')
    assert isinstance(ls.LensModel.lens_model, types.ModuleType)

    # Symbols inside the module are available
    assert 'LensModel' in ls.LensModel.lens_model.__all__
    assert hasattr(ls.LensModel.lens_model, 'LensModel')
    assert isinstance(ls.LensModel.lens_model.LensModel, type)


def test_laconic():
    _remove_cached_lenstronomy()
    import lenstronomy as ls

    # Without running .laconic(), nothing is changed:
    assert not hasattr(ls, 'LensModel')
    assert not hasattr(ls, 'MultiBandImageReconstruction')

    ls.laconic()

    # We can access lenstronomy symbols directy
    assert hasattr(ls, 'MultiBandImageReconstruction')
    assert isinstance(ls.MultiBandImageReconstruction, type)

    # Non-lenstronomy symbols are not accessible
    assert not hasattr(ls, 'np')
    assert not hasattr(ls, 'numpy')

    # Clashing symbols are not accessible
    assert not hasattr(ls, 'PSF')

    # 'short' type access still works
    assert hasattr(ls, 'Analysis')
    assert isinstance(ls.Analysis, types.ModuleType)
    assert hasattr(ls.Analysis, 'image_reconstruction')
    assert isinstance(ls.Analysis.image_reconstruction, types.ModuleType)
    assert hasattr(ls.Analysis.image_reconstruction,
                   'MultiBandImageReconstruction')
    assert isinstance(ls.Analysis.image_reconstruction.MultiBandImageReconstruction,
                      type)

    # Key classes clashing with submodule names are
    # accessible with an extra underscore
    assert hasattr(ls, 'LensModel')
    assert isinstance(ls.LensModel, types.ModuleType)
    assert hasattr(ls, 'LensModel_')
    assert isinstance(ls.LensModel_, type)
