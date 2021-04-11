# -*- coding: utf-8 -*-
#
import time
import sys
import os

sys.path.insert(0, os.path.abspath("../"))

extensions = [
    #    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
#    "sphinx.ext.githubpages", # we don't need this because we will manually place CNAME and .nojekyll
#    "sphinx.ext.coverage", # don't need this either. We will shift to docstr-coverage
    "sphinx_multiversion",
    "m2r",
]
source_suffix = [".rst", ".md"]
html_last_updated_fmt = "%c"
master_doc = "index"
project = "Box Embeddings"
copyright = "2021, Information Extraction and Synthesis Lab, UMass"
exclude_patterns = ["_build", "**/docs", "**/.docs"]

pygments_style = "friendly"
html_theme = "alabaster"
def setup(app):
  app.add_css_file( "custom_t.css" )
templates_path = ["templates"]  # needed for multiversion
#autoclass_content = "class"

html_baseurl = "http://iesl.cs.umass.edu/box-embeddings/"
html_logo = "images/UMass_IESL.png"
html_theme_options = {
    "github_user": "iesl",
    "github_repo": "box-embeddings",
    "github_banner": True,
    "github_button": True,
    #"description": "Python implementation for box embeddings and box representations",
}
html_css_files = ['custom_t.css']

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"
# autodoc_default_options = {'undoc-members': True}

# API Generation
autoapi_dirs = ["../box_embeddings"]
autoapi_root = "."
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    #"show-module-summary",
]
autoapi_add_toctree_entry = False
autoapi_keep_files = True

# see: https://github.com/data-describe/data-describe/blob/master/docs/source/conf.py
# and https://github.com/data-describe/data-describe/blob/master/docs/make.py
# Multiversioning
smv_tag_whitelist = r"^v\d+\.\d+\.\d+b?\d*$"
smv_branch_whitelist = r"^.*main$"
smv_remote_whitelist = r"^.*$"
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "versioning.html",
    ]
}
