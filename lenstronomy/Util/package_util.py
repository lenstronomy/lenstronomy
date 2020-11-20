import types


def exporter(export_self=False):
    """Export utility, modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def laconic():
    """Activate laconic / super-shortcut mode.

    Usage:
        import lenstronomy as ls
        ls.laconic()

        lens_model = ls.LensModel(...)
        source_model = ls.LightModel(...)
        image_model = ls.ImageModel(...)
    """
    short(_laconic=True)


@export
def short(_laconic=False):
    """Activate shortcut mode.

    Usage:
        import lenstronomy as ls
        ls.short()

        lens_model = ls.LensModel.lens_model.LensModel(...)
        source_model = ls.LightModel.light_model.LightModel(...)
        image_model = ls.ImSim.image_model.ImageModel(...)
    """
    import pkgutil
    import lenstronomy

    to_add = dict()
    all_modules = dict()

    for loader, module_name, is_pkg in pkgutil.walk_packages(lenstronomy.__path__):
        # Load the module
        module = all_modules[module_name] = \
            loader.find_module(module_name).load_module(module_name)

        if '.' in module_name:
            # Submodule, e.g. Data.psf
            # Monkeypatch the parent module to make it accessible
            fragments = module_name.split('.')
            parent_module_name, child_name = '.'.join(fragments[:-1]), fragments[-1]
            setattr(all_modules[parent_module_name], child_name, module)
        else:
            # Top-level module, e.g. Data: add as lenstronomy attribute
            to_add[module_name] = module

        if _laconic:
            # If the module defines an __all__, load its symbols as well.
            # (unlike import *, we do not just load everything if __all__ is missing)
            if hasattr(module, '__all__'):
                for symbol in module.__all__:
                    symbol_name = symbol
                    if isinstance(to_add.get(symbol), types.ModuleType):
                        # Key class clashing with module name
                        # (Cosmo, LensModel, LightModel, PointSource)
                        # Try to add the symbol as LensModel_:
                        symbol_name = symbol + '_'
                    if symbol_name in to_add:
                        # Name clash! Do not add the symbol
                        # or the one it clashed with.
                        del to_add[symbol_name]
                    else:
                        to_add[symbol_name] = getattr(module, symbol)

    # Make symbols accessible under lenstronomy.[x]
    for key, value in to_add.items():
        setattr(lenstronomy, key, value)
    lenstronomy.__all__ = to_add.keys()
