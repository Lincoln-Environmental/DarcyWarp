.. include:: ../docs/2D_RUNTIME_2026-11-12.rst


Prescribed-head budget rates
----------------------------

The 2D water budget reports ``chd_in`` and ``chd_out`` using the MODFLOW 6
CHD convention. Each prescribed-head cell contributes one signed rate, the
sum of its flows to all neighbouring active cells. Positive cell rates
contribute to inflow and negative cell rates to outflow. Adjacent prescribed
heads can exchange water directly. A prescribed-head cell can also pass water
between two free cells and have zero net package exchange.

Use ``free_chd_in`` and ``free_chd_out`` to measure gross exchange across
the faces between prescribed-head and free cells. The transient solver uses
the free-cell water balance when checking the accepted head. This keeps
exchange between prescribed heads from masking a storage error elsewhere.
