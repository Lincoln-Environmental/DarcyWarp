# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DarcyWarp'
copyright = '2026, Patrick Durney and Jaco Fourie'
author = 'Patrick Durney and Jaco Fourie'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = []
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_title = 'DarcyWarp | Groundwater modelling'
html_theme_options = {
    'description': 'Structured 2D and 3D groundwater flow on CPUs and NVIDIA GPUs.',
    'github_user': 'Lincoln-Environmental',
    'github_repo': 'DarcyWarp',
    'github_button': False,
    'extra_nav_links': {
        'Source code': 'https://github.com/Lincoln-Environmental/DarcyWarp',
        'Report an issue': 'https://github.com/Lincoln-Environmental/DarcyWarp/issues',
    },
    'font_family': 'Arial, Helvetica, sans-serif',
    'head_font_family': 'Arial, Helvetica, sans-serif',
    'body_text': '#263c46',
    'sidebar_header': '#123a4a',
    'sidebar_link': '#176a76',
    'link': '#176a76',
    'link_hover': '#123a4a',
    'code_bg': '#eef5f6',
    'note_bg': '#eef5f6',
    'note_border': '#86b6be',
    'page_width': '1120px',
    'sidebar_width': '240px',
}
