# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
PYTHON        ?= python3
# Use the Python module invocation so the Makefile works even when
# the `sphinx-build` console script is not on PATH.
SPHINXBUILD   ?= $(PYTHON) -m sphinx
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(PYTHON) -c "import sphinx" >/dev/null 2>&1 || { \
		echo "Sphinx is not installed. Run: conda env update -f environment.yml" >&2; \
		exit 1; \
	}
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help docs docs-clean Makefile

# Friendly aliases for the common documentation workflow.
docs: html

docs-clean: clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(PYTHON) -c "import sphinx" >/dev/null 2>&1 || { \
		echo "Sphinx is not installed. Run: conda env update -f environment.yml" >&2; \
		exit 1; \
	}
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
