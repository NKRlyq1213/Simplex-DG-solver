# Simplex-DG-solver

Simplex-DG-solver is a research-oriented discontinuous Galerkin solver for
advection on triangular elements, with a Gaussian transport example on the
sphere in `examples/step9_gaussian_convergence.py`.

The current branch includes three SBP-compatible reference/operator variants:

- `projected`
- `full-raw`
- `full-orth`

`projected` is the original projected-SBP workflow. `full-raw` and
`full-orth` use Table 1 direct boundary extraction and full-SBP-compatible
lifting. `full-raw` and `full-orth` are algebraically equivalent
constructions of the same corrected differentiation operators.

## Requirements

- Python
- NumPy
- SciPy
- Matplotlib
- Numba (optional at runtime, but included in `requirements.txt`)
- Pytest for the test suite

Install the dependencies with:

```bash
python -m pip install -r requirements-dev.txt
```

## Repository Layout

- `src/simplex_dg/`
  Core reference, geometry, trace, RHS, diagnostics, and time-integration
  code.
- `examples/step9_gaussian_convergence.py`
  The main Gaussian transport runner on the sphere.
- `scripts/run_task5_full_sbp_validation.py`
  Validation orchestrator for projected vs full-SBP comparisons.
- `tests/`
  Regression and validation tests for projected and full-SBP workflows.
- `results/task5_full_sbp_validation/`
  Generated Task 5 experiment outputs, summaries, and plots.

## Running the Test Suite

Run the full suite with:

```bash
pytest -q
```

The test suite covers:

- projected-workflow regression protection
- Table 1 direct boundary extraction
- Table 1 full-SBP operator construction
- projected/full cache integration
- projected/full surface integration
- Step9 CLI propagation for `--sbp`
- Task 5 validation helpers

## Step9 Gaussian Convergence Runner

The Step9 example is the single Gaussian convergence runner for all supported
SBP variants.

Basic usage:

```bash
python examples/step9_gaussian_convergence.py --help
```

Typical run:

```bash
python examples/step9_gaussian_convergence.py \
  --ndivs 4 8 16 32 \
  --order 4 \
  --table table1 \
  --sbp full-orth \
  --form split \
  --flux upwind
```

### `--sbp` Variants

Step9 accepts:

- `--sbp projected`
- `--sbp full-raw`
- `--sbp full-orth`

Default:

```bash
--sbp projected
```

Variant meaning:

- `projected`
  Existing projected-SBP differentiation, projected trace, and polynomial
  lift.
- `full-raw`
  Table 1 full-SBP operator constructed in the raw basis, with direct
  boundary extraction and `H^{-1} E^T W_b` lifting.
- `full-orth`
  Algebraically equivalent discretely orthogonalized full-SBP construction.

Compatibility matrix:

| table  | projected | full-raw | full-orth |
| ------ | --------: | -------: | --------: |
| table1 |   allowed |  allowed |   allowed |
| table2 |   allowed | rejected |  rejected |

For full variants, Step9 validates the restriction before running. The full
variants are tied to Table 1 because they use direct extraction of Table 1
boundary volume nodes.

### Important CLI Options

- `--ndivs`
  Mesh refinement levels.
- `--order`
  Polynomial order.
- `--table`
  Triangle quadrature table: `table1` or `table2`.
- `--sbp`
  `projected`, `full-raw`, or `full-orth`.
- `--form`
  Volume form: `conservative` or `split`.
- `--flux`
  Surface flux: `central`, `upwind`, or `lf`.
- `--cfl`
  CFL number.
- `--tf`
  Final time.
- `--history-every`
  History sampling stride.
- `--output`
  Base CSV output path.
- `--plot-dir`
  Base plot directory.
- `--no-plots`
  Disable plot generation.
- `--no-numba`
  Use the pure NumPy path.

### Output Naming

Step9 appends the scheme identifier to avoid collisions across variants.

For example, with:

- `table1`
- `full-orth`
- `conservative`
- `central`

the default output names become:

```text
outputs/convergence/gaussian_table1_full-orth_conservative_central.csv
outputs/convergence/gaussian_table1_full-orth_conservative_central_metadata.json
outputs/convergence/plots/table1_full-orth_conservative_central/
```

The metadata JSON includes `sbp_variant` and the full scheme identifier.

## Full-SBP Reference Components

Key reference-layer modules:

- `src/simplex_dg/reference/sbp_variants.py`
  Shared variant naming and mapping helpers.
- `src/simplex_dg/reference/table1_boundary.py`
  Table 1 direct boundary extraction data:
  `face_indices`, `face_extract`, `face_weights`, `Br`, and `Bs`.
- `src/simplex_dg/reference/table1_full_sbp.py`
  Full-SBP corrected differentiation operators for `raw` and
  `orthogonalized` constructions.

`build_reference_cache(...)` now accepts:

```python
build_reference_cache(order=4, table="table1", sbp_variant="projected")
build_reference_cache(order=4, table="table1", sbp_variant="full-raw")
build_reference_cache(order=4, table="table1", sbp_variant="full-orth")
```

The variant atomically selects:

- `Dr`, `Ds`
- face trace operator
- face lift operator
- boundary representation metadata

## Task 5 Validation Script

The repository includes a validation orchestrator for PDE-level comparisons:

```bash
python scripts/run_task5_full_sbp_validation.py --help
```

Supported phases:

- `equivalence`
- `mass`
- `convergence`
- `stability`
- `product-rule`
- `all`

Example:

```bash
python scripts/run_task5_full_sbp_validation.py \
  --phase equivalence \
  --output-dir results/task5_full_sbp_validation
```

The default output layout is:

```text
results/task5_full_sbp_validation/
├── raw_orth_equivalence/
├── mass/
├── convergence/
├── stability/
├── product_rule/
├── summary.csv
├── summary.json
└── report.md
```

## Current Validation Snapshot

The current Task 5 outputs were generated with the existing numerical method
unchanged. Highlights from `results/task5_full_sbp_validation/summary.json`:

- `full-raw` and `full-orth` match to floating-point roundoff in the tested
  multi-step runs.
- projected and full-SBP variants both remain finite in the tested short- and
  finite-time Step9 runs.
- order-4 Table 1 runs show approximately fifth-order relative `L2` error
  convergence on the tested mesh sequence.
- full-SBP reduces the discrete product-rule residual constant on the tested
  cases, but does not change it into a different asymptotic order in the
  current measurements.

See:

- `results/task5_full_sbp_validation/report.md`
- `results/task5_full_sbp_validation/summary.json`

## Known Scope Limit

The nonzero constant-state RHS was intentionally not investigated or modified
in the current validation task. This branch does not claim free-stream
preservation from these recent changes.

## License

See `LICENSE`.
