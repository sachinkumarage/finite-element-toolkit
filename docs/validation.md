# FEA Validation (Version 29)

Infrastructure for comparing simulation results against trusted
reference or experimental data. This document is the detailed
validation guide; see [`docs/verification.md`](verification.md) for the
verification guide (a different, purely mathematical question) and the
main [README](../README.md#version-29) for a shorter overview.

## Validation vs. verification

> **Verification**: "Are we solving the equations correctly?" --
> answered against a *known* reference (an analytical formula, a mesh
> refinement trend, another solver). No external data needed.
>
> **Validation**: "Are we solving the correct physical problem?" --
> answerable only against *external, trusted evidence*: experimental
> measurement, published reference data, or a prior high-fidelity
> simulation.

`femtoolkit.validation` never blurs this line. It ships with **no
bundled dataset** and performs **no validation of its own** -- a
comparison only happens when a caller supplies real reference data
through `ReferenceDataset`. A model this toolkit has not been given
reference data for is *unvalidated*, and every report this toolkit
generates says so explicitly (see
[`docs/reporting.md`](reporting.md)'s "Validation Results" section,
which reads *"No reference/experimental dataset was supplied for this
report -- this model has not been validated against external data"*
when `validation_results` is empty).

## Reference datasets

`femtoolkit.validation.datasets.ReferenceDataset` is a structured,
metadata-carrying container:

```python
from femtoolkit.validation import ReferenceDataset
import numpy as np

dataset = ReferenceDataset(
    name="Cantilever deflection profile",
    source="Roark's Formulas for Stress and Strain, 8th ed., Table 8.1",
    quantity="Deflection",
    units="m",
    independent_variable=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    independent_variable_label="x (m)",
    values=np.array([0.0, -1.7e-4, -6.4e-4, -1.3e-3, -2.1e-3]),
    uncertainty=None,          # optional per-point uncertainty
    description="Measured deflection profile from the referenced handbook.",
)
```

`source` is a **required, non-blank field** -- every dataset must say
where it came from. For a dataset built from this toolkit's own
closed-form formulas (useful for demonstrating the comparison machinery
without real external data, as
`examples/validation/reference_dataset_example.py` does), the
convention used throughout this toolkit's own examples and tests is an
explicit `source="self-generated: <formula description>"` prefix --
never a source string that could be mistaken for an experimental
citation.

### Loading from a file

```python
from femtoolkit.validation import load_reference_dataset_csv, load_reference_dataset_json

# CSV: header row, then independent_variable,value[,uncertainty] rows
dataset = load_reference_dataset_csv(
    "beam_deflection.csv", name="...", source="...", quantity="...", units="m"
)

# JSON: a dict matching ReferenceDataset's fields
dataset = load_reference_dataset_json("beam_deflection.json")
```

Both loaders raise `ValidationError` (not a bare `FileNotFoundError` or
`json.JSONDecodeError`) for a missing file, malformed JSON, a
non-numeric CSV value, or a missing required field -- consistent,
catchable error handling regardless of format (spec section 22: "handle
missing reference data gracefully").

## Comparing simulation results

```python
from femtoolkit.validation import compare_to_reference_dataset

result = compare_to_reference_dataset(dataset, simulation_values)
print(result.status, result.message)
```

`compare_to_reference_dataset` performs **no interpolation or
resampling** -- `simulation_values` must already be aligned with
`dataset.values` (same length, same order, same units); this is a
deliberate simplicity choice, not an oversight, since silently
resampling would hide a real alignment bug. It reuses the same
L2/relative-L2 error metrics `femtoolkit.verification.metrics` uses
(validation and verification ask different questions, but "how wrong is
this number" is computed the same way either time), and the default
tolerance (`Tolerance(absolute=1e-6, relative=0.05)`) is deliberately
**looser** than the verification benchmarks' default
(`relative=1e-6`): a reference dataset -- especially an experimental one
-- carries its own measurement or discretization uncertainty, so
demanding floating-point agreement from a validation comparison would
be a category error.

## Uncertainty

`ReferenceDataset.uncertainty` records a per-point uncertainty when the
source dataset reports one; `ValidationResult.has_uncertainty` reflects
whether it was present. **Version 29 does not propagate this
uncertainty into the tolerance check** -- `compare_to_reference_dataset`
uses the same fixed `Tolerance` regardless of whether `uncertainty` is
set. A future version could widen the effective tolerance by the
reported uncertainty at each point; this version reports the
uncertainty's presence for the caller's own judgment but does not act
on it automatically, to avoid silently loosening a comparison in a way
that is not yet carefully justified.

## Never fabricate experimental data

This is enforced by construction, not just documentation:

- `femtoolkit.validation` contains no data files, no example
  measurements, no "typical" reference values -- only the container
  type and the comparison function.
- Every example that demonstrates the comparison machinery
  (`examples/validation/reference_dataset_example.py`) builds its
  dataset from a closed-form formula and labels the source
  `"self-generated: ..."`, explicitly stating in its own module
  docstring that this is *not* a validation claim.
- The GUI's Verification & Validation page only ever compares against a
  dataset the user explicitly uploads (`femtoolkit.gui.workflow_pages.verification_page`)
  -- there is no "run validation" button that does not require a file.
- The engineering report renderer (`femtoolkit.reporting.renderers.render_markdown`)
  prints an explicit "not validated" statement when no validation
  results are present, rather than omitting the section silently.

## Limitations

No bundled reference/experimental dataset ships with this toolkit (by
design -- see above). No automatic interpolation/resampling between a
dataset's independent-variable grid and a simulation's own grid. No
uncertainty-aware tolerance widening (see above). No support for
multi-dimensional reference fields (e.g. a full 2D contour) -- only
1D series indexed by a single independent variable.
