API reference
=============

Model construction
------------------

.. currentmodule:: DARCY_WARP_PACKAGE

.. autofunction:: create_solver

Ensemble benchmark helpers
--------------------------

.. currentmodule:: DARCY_WARP_PACKAGE.benchmark_support

.. autofunction:: default_workers_arg

.. autofunction:: effective_worker_count

.. autofunction:: write_metadata_if_enabled

Storage relationships
---------------------

.. currentmodule:: DARCY_WARP_PACKAGE.physics.storage_2d

.. autofunction:: specific_storage_potential

.. autofunction:: secant_specific_yield_coeff

.. autofunction:: secant_specific_storage_coeff

.. autofunction:: exact_unconfined_storage_terms

Operator and boundary relationships
----------------------------------

.. currentmodule:: DARCY_WARP_PACKAGE.physics.operator_data

.. autofunction:: normalize_scalar_or_grid_to_shape

.. autofunction:: compute_ghb_factor_from_raw_fields

.. currentmodule:: DARCY_WARP_PACKAGE.physics.drn_2d

.. autofunction:: drn_discharge_2d

.. autofunction:: add_drn_to_budget

.. currentmodule:: DARCY_WARP_PACKAGE.physics.riv_2d

.. autofunction:: riv_discharge_2d

.. currentmodule:: DARCY_WARP_PACKAGE.physics.budgets_2d

.. autofunction:: compute_mass_balance_budget

Transient simulation
--------------------

.. currentmodule:: DARCY_WARP_PACKAGE.solvers.transient_unconfined

.. autofunction:: solve_transient_unconfined_backend

Benchmark budgets
-----------------

.. currentmodule:: DARCY_WARP_PACKAGE.process_mass_balance

.. autofunction:: compare_mass_balance
