# Examples

## Steady recharge in a confined aquifer

From the repository root, with the DarcyWarp environment activated:

```bash
python -m examples.steady_recharge
```

This example runs on the **CPU** and needs no external data or MODFLOW executable.
It constructs a 41 × 17 grid with 25 m square cells, uniform transmissivity
of 0.001 m²/s, and recharge of 10⁻⁸ m/s. The west and east boundary cells
have a fixed head of 10 m. The north and south outer faces have no flow.
The analytical distance between the fixed-head cell centres is 1,000 m.

The calculated head is checked against

$$h(x) = h_0 + \frac{R}{2T}x(L-x),$$

where $h_0$ is the fixed head, $R$ is recharge, $T$ is transmissivity,
and $L$ is the distance between the fixed-head cell centres. The expected
maximum head is **11.25 m**. The script requires solver convergence,
an absolute head error below 10⁻⁵ m and a relative water-balance error
below 10⁻⁵. These checks concern this small homogeneous confined case.

Results are written to `outputs/steady_recharge/`:

| File | Contents |
| --- | --- |
| `steady_recharge.npz` | Input arrays, calculated and analytical heads, coordinates and error metrics; names include physical units. |
| `steady_recharge.png` | Hydraulic-head map and centreline comparison. |
| `steady_recharge.svg` | The same figure for documentation and vector export. |

The example uses metres and seconds throughout. Recharge is zero on the
fixed-head cells because their heads are prescribed rather than calculated
from a recharge balance. Net recharge into the remaining cells is compared
with discharge to the fixed-head cells.

Edit the explicit parameters in the `main(...)` call at the bottom of
[`steady_recharge.py`](steady_recharge.py), or call it from Python:

```python
from examples.steady_recharge import main

summary = main(device="cpu", output_dir="outputs/my_first_model", make_plot=True)
```

Use `device="cuda:0"` when a supported NVIDIA GPU and driver are available.
The example explicitly requires `DARCY_FLOAT=float64`; run it in a fresh
Python process when changing precision. Set `make_plot=False` if only the
numerical arrays and checks are needed. The first run compiles Warp kernels.

For the installation steps, input conventions and next steps, see
[`source/quickstart.rst`](../source/quickstart.rst). Benchmark and verification
workflows are documented separately from this introductory example.
