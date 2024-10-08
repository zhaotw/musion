import pkgutil
import importlib
import os

import musion

task_names = [module.name for module in pkgutil.iter_modules(musion.__path__)]
task_names.remove('util')

def get_task_instance(task_name: str, **init_kwargs) -> musion.MusionBase:
    musion_pkg = importlib.import_module('musion.' + task_name)
    musion_task = getattr(musion_pkg, dir(musion_pkg)[0])(**init_kwargs)

    return musion_task

def get_task_description(task_name: str):
    musion_pkg = importlib.import_module('musion.' + task_name)
    readme_path = os.path.join(musion_pkg.__path__[0], "README.md")
    with open(readme_path, 'r') as f:
        description = f.readlines()[2]
    return description