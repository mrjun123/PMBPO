import importlib

def get_item(name):
    dict = {
        "dropout_mbpo_model": "DropoutBbpoModel"
    }
    module = importlib.import_module("mymbrl.models."+name)
    module_class = getattr(module, dict[name])
    return module_class
    