import sys
from unittest.mock import MagicMock

MOCK_MODULES = [
    'pyvista', 
    'vtk', 
    'vtkmodules', 
    'vtkmodules.numpy_interface', 
    'vtkmodules.numpy_interface.dataset_adapter'
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

project = 'HomiCSx'
copyright = '2026, Amir Reza Tahouni'
author = 'Amir Reza Tahouni'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx_rtd_theme',
    # 'sphinx.ext.mathjax',
    'sphinxcontrib.katex',
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    # "deflist",
    # "colon_fence",
]

# katex_options = {
#     'strict': False,
#     'output': 'html',
#     'displayMode': True,
# }

# katex_prerender = True
# myst_dmath_double_inline = True
# myst_update_mathjax = False

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autosummary_generate = True

templates_path = ['_templates/autosummary']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Path setup --------------------------------------------------------------
import os

sys.path.insert(0, os.path.abspath('../../'))

autodoc_mock_imports = [
    "numpy", 
    "pandas", 
    "torch", 
    "scipy", 
    "dolfinx", 
    "dolfinx.fem", 
    "dolfinx.mesh", 
    "dolfinx_mpc", 
    "paraview", 
    "mpi4py", 
    "ufl", 
    "basix", 
    "petsc4py", 
    "matplotlib",
    "matplotlib.pyplot",
    "gmsh",
    "meshio",
]

autodoc_default_options = {
    'members': True,
    'private-members': False,
    'special-members': False,
    'inherited-members': False,
    'member-order': 'bysource',
    'undoc-members': False,
    'show-inheritance': False,
    'exclude-members': '__weakref__'
}

napoleon_use_ivar = True
napoleon_use_param = False
napoleon_preprocess_types = True
napoleon_attr_annotations = True
autodoc_typehints = "none"


import inspect
def skip_members(app, what, name, obj, skip, options):
    if name.startswith('__'):
        return True

    if name.startswith('_'):
        return True

    if what == "class":
        if isinstance(obj, property):
            return False

        if inspect.isroutine(obj):
            return False

        return True

    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_members)

