import pkgutil
import importlib
import os
import inspect

# Dynamically import all modules and classes in the current package
__all__ = []  # To control what gets exposed with `from subpackage import *`

package_dir = os.path.dirname(__file__)  # Get the current directory

for loader, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    # Import the module
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module
    __all__.append(module_name)

    # Add all classes from the module to the globals
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Ensure the class is defined in this module
        if obj.__module__ == module.__name__:
            globals()[name] = obj
            __all__.append(name)