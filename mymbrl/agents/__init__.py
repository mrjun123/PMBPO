import importlib

def get_item(name):
    dict = {
        "dropout_mbpo": "DropoutMbpo"
    }
    module = importlib.import_module("mymbrl.agents."+name)
    module_class = getattr(module, dict[name])
    return module_class
    # return dict[name]