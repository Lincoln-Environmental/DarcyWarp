Build the documentation
=======================

The documentation uses Sphinx and the sources in ``source/``. Sphinx is
included in ``environment.yml``. A documentation-only environment can instead
install it with:

.. code-block:: bash

   python -m pip install sphinx

Build HTML
----------

From the repository root:

.. code-block:: bash

   python -m sphinx -b html source build/html

Open ``build/html/index.html`` in a browser. This direct command works without
Make and uses the active Python environment.

The root Makefile provides equivalent convenience targets:

.. code-block:: bash

   make docs
   make docs-clean docs

``make docs`` aliases the HTML build, and ``make docs-clean`` removes generated
Sphinx output. Use ``make help`` to list other targets, or
``make docs PYTHON=python`` to choose the Python executable.

Check a documentation change
----------------------------

Build with warnings treated as errors:

.. code-block:: bash

   python -m sphinx -b html -n -W --keep-going source build/html

Review the rendered page as well as the build log, especially tables, code
blocks, images and navigation. Sphinx checks internal references during the
build; external URLs need a separate check when they change.

Generated files in ``build/`` are local outputs. Commit the documentation
sources and intentional source assets, rather than a generated HTML tree.

Troubleshooting
---------------

If Python cannot import Sphinx, install it into the environment used for the
build. For the project conda environment:

.. code-block:: bash

   conda activate darcywarp
   conda env update -f environment.yml

If a Make invocation selects the wrong Python interpreter, use the direct
``python -m sphinx`` command above.
