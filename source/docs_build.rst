Documentation Build
===================

Sphinx docs in this repository are built from the ``source/`` directory using
the root ``Makefile``.

Make targets
------------

Common targets:

* ``make help``: list Sphinx make-mode targets
* ``make docs``: build HTML documentation (alias for ``make html``)
* ``make docs-clean``: remove generated build output (alias for ``make clean``)

Linux helper script
-------------------

For a reproducible Linux workflow, use:

.. code-block:: bash

   bash scripts/generate_docs.sh

What the script does:

* verifies ``make`` and ``python3`` are installed
* verifies Sphinx is importable in the current Python environment
* runs ``make docs-clean`` (unless ``--no-clean`` is passed)
* runs the selected target (default: ``docs``)

Examples:

.. code-block:: bash

   bash scripts/generate_docs.sh
   bash scripts/generate_docs.sh --no-clean
   bash scripts/generate_docs.sh latexpdf

Output location
---------------

HTML output is written to:

.. code-block:: text

   build/html/index.html

Troubleshooting
---------------

If you see ``Sphinx is not installed``, update the conda environment:

.. code-block:: bash

   conda env update -f environment.yml
