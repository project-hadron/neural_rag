# Configuration file for the Sphinx documentation builder.
import os
import sys
from nn_rag import __version__

sys.path.insert(0, os.path.abspath('../'))

# -- Project information
project = 'RAII RAG'
copyright = '2024, gigas64'
author = 'gigas64'

# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
html_static_path = ['_static']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

exclude_patterns = ['.DS_Store', '.ipynd_checkpoints', ]
