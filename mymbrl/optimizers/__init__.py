import importlib

def get_item(name):
    dict = {
    }
    module = importlib.import_module("mymbrl.optimizers."+name)
    module_class = getattr(module, dict[name])
    return module_class