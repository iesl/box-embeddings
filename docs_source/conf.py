# -*- coding: utf-8 -*-
#

import sys
import os

sys.path.insert(0, os.path.abspath("../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    "m2r",
]
source_suffix = [".rst", ".md"]
# source_suffix = '.rst'
master_doc = "index"
project = "Box Embeddings"
copyright = "Information Extraction and Synthesis Lab., UMass"
exclude_patterns = ["_build", "**/docs", "**/.docs"]
pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"
autoclass_content = "class"
html_baseurl = "http://iesl.cs.umass.edu/box-embeddings/"
html_logo = "images/UMass_IESL.png"

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"
# autodoc_default_options = {'undoc-members': True}
