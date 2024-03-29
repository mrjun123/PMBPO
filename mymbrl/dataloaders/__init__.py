import importlib

def get_item(name):
    dict = {
        "dmbpo": "DMBPO"
    }
    module = importlib.import_module("mymbrl.dataloaders."+name)
    module_class = getattr(module, dict[name])
    return module_class