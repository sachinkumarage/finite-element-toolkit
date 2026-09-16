# Engineering GUI & Interactive Simulation Workspace (Version 24)

A Streamlit-based application layer over the existing finite element
toolkit: `Project -> Model -> Material -> Mesh -> Boundary Conditions ->
Loads -> Solver -> Run -> Results -> Visualization`. This document is
the detailed operator's guide for that workspace; see the main
[README](../README.md#version-24) for a shorter overview and how
Version 24 fits into the toolkit's history.

## Architecture

Three layers, one direction of dependency:

```text
femtoolkit.gui              Streamlit widgets only
     |
femtoolkit.application      service layer -- no Streamlit import
     |
femtoolkit core              materials, mesh, analysis, thermal, postprocessing
```

Nothing in `femtoolkit.gui` or `femtoolkit.application` assembles a
stiffness matrix, integrates a shape function, or solves a linear
system. `femtoolkit.application` only builds and validates plain
configuration and orchestrates the existing, unmodified solvers
(`StaticLinearAnalysis`, `SteadyStateThermalAnalysis`); `femtoolkit.gui`
only renders Streamlit widgets and delegates every action to a service
class. This separation is checked by `tests/test_gui_no_streamlit.py`:
the entire application layer, and every part of `femtoolkit.gui` except
`app.py`/the page modules, imports and works correctly even with
Streamlit uninstalled.

## The application/service layer

`src/femtoolkit/application/`:

- **`project.py`** -- the plain, JSON-serializable configuration
  dataclasses: `Project`, `MaterialConfig`, `MeshConfig`,
  `BoundaryConditionConfig`, `LoadConfig`, `SolverConfig`. None of these
  ever holds a live `Mesh`, material, or analysis object -- only the
  configuration describing how to build one.
- **`analysis_types.py`** -- the registry of analysis types the GUI
  offers (`SUPPORTED_ANALYSIS_TYPES`), each marked `available=True/False`
  with an honest `reason` when not yet wired into this version's GUI
  workflow. See "Supported analysis types" below.
- **`materials_catalog.py`** -- a handful of preset engineering
  materials (Structural Steel, Aluminum 6061-T6, Copper, Titanium
  Ti-6Al-4V) plus a `"custom"` entry, with standard textbook
  room-temperature property values.
- **`validation.py`** -- `validate_project` and its per-category
  sub-validators (`validate_material`, `validate_mesh`,
  `validate_boundary_conditions`, `validate_loads`, `validate_solver`),
  producing a `ValidationResult` with human-readable error messages.
  Never constructs a real mesh or material -- pure, fast, structural
  checks on the plain `Project` data.
- **`model_service.py`** -- `ModelService`, the one place that touches
  real toolkit objects: `build_mesh` (via the Version 8 structured
  generator), `build_thermal_materials`, `build_boundary_conditions`/
  `build_loads` (resolving a region name to nodes via
  `Mesh.nodes_on_boundary`, Version 9, then building the existing
  `BoundaryCondition`/`NodalLoad`/`PrescribedTemperature`/
  `PrescribedHeatFlux` objects), and `mesh_summary`.
- **`simulation_service.py`** -- `SimulationService.run(project)`:
  validate first (refusing to solve at all if invalid), then build,
  then solve through the *existing, unmodified* solver, then wrap
  through the Version 22 post-processing pipeline
  (`from_static_linear`/`from_thermal_steady_state` +
  `with_derived_fields`). Returns a `SimulationRunResult` with a
  `status` of `"completed"`, `"invalid"`, or `"failed"`.
- **`results_service.py`** -- `ResultsService`: `summary` (the Version
  22 `EngineeringSummary`), `available_nodal_fields`/
  `available_element_fields` (only what the result actually carries),
  `field_range`.
- **`project_service.py`** -- `ProjectService`: create/save/load/reset,
  JSON serialization.
- **`exceptions_display.py`** -- `describe_error`/
  `describe_visualization_error`: map a caught exception to a short,
  professional, user-facing message (never a raw traceback).

## The project model and serialization

A `Project` is saved as plain JSON via `ProjectService.to_json`/
`from_json`/`save`/`load`:

```json
{
  "name": "Cantilever Beam",
  "analysis_type": "linear_static",
  "material": {"name": "Structural Steel", "youngs_modulus": 2e11, ...},
  "mesh": {"width": 2.0, "height": 0.4, "nx": 16, "ny": 4, ...},
  "boundary_conditions": [{"region": "left", "dof": "X", "value": 0.0}, ...],
  "loads": [{"region": "right", "dof": "Y", "magnitude": -2000.0}],
  "solver": {"tolerance": 1e-6, "max_iterations": 25},
  "created_at": "2026-01-01T00:00:00+00:00",
  "format_version": 1
}
```

No solver object, mesh, or NumPy array is ever serialized -- only the
configuration needed to rebuild one. `format_version` lets a future
version detect and migrate an older project file; unknown top-level
keys in a loaded file are ignored, so a project saved by a newer
version can still be partially read by this one.

## Supported analysis types

| Key | Label | Available |
| --- | --- | --- |
| `linear_static` | Linear Static (Mechanical) | Yes |
| `thermal_steady_state` | Steady-State Thermal | Yes |
| `nonlinear_static` | Nonlinear Static (Mechanical) | No -- supported by the core toolkit since Version 13, not yet wired into this GUI's project/material/load-stepping model |
| `thermomechanical` | Sequential Thermomechanical | No -- supported by the core toolkit (Versions 19-21), not yet wired into this GUI's combined workflow |

The GUI's Project page marks the two unavailable types with their
reason rather than pretending to run them -- no unsupported physics is
ever silently simulated.

## Streamlit state management

Streamlit reruns the whole script on every interaction. `AppState`
(`src/femtoolkit/gui/state.py`) wraps `st.session_state` behind a small
typed interface (`project`, `last_run`, `visualization_settings`,
`reset()`, `has_project()`, `has_results()`) so no page touches
`st.session_state` directly. `AppState` takes any `MutableMapping` in
its constructor -- in the running app that's `st.session_state`, in
tests it's a plain `dict` -- which is what makes state transitions
testable without ever starting a Streamlit server. Setting a new
project automatically clears any previously computed simulation result,
so a stale result can never be shown against a changed model.

## The simulation workflow

```text
Validate Model
      |
Run Simulation
      |
Simulation Status
      |
Results Available
```

The Run page validates first and disables the "Run Simulation" button
if the project is invalid. A run that fails validation never reaches
the solver at all; a run whose solver genuinely raises (a
`FiniteElementToolkitError` subclass, or any unexpected exception) is
caught and reported as a "Solver error"/"Unexpected error" message, not
a traceback.

## Material configuration

The Material page lets you apply a catalog preset or edit properties
directly, with unit labels and immediate validation feedback:
`Young's modulus > 0`, `-1 < Poisson's ratio < 0.5`, `thermal
conductivity > 0`, `specific heat > 0` (the two thermal checks only
apply to a thermal analysis; the two mechanical checks only apply to a
mechanical one).

## Mesh inspection

The Mesh page configures a structured 2D domain (width, height,
subdivisions, Q4 or CST elements, thickness for a mechanical analysis)
via the existing `create_quad_mesh`/`create_triangular_mesh` generators
(Version 8), then reports node/element counts, dimension, element type,
and the Version 8 shape-quality summary (`min`/`max`/`average`
quality, invalid element count), plus a Matplotlib node/edge preview.
This version does not add a 3D structured generator or any CAD
capability -- see the Version 25 preview.

## Boundary conditions and loads

Both are region-based: pick a named boundary (`left`/`right`/`top`/
`bottom`, from the Version 9 `Rectangle`/`BoundaryRegion` machinery),
a DOF (`X`/`Y` for mechanical, `TEMPERATURE`/`HEAT_FLUX` for thermal),
and a value. The configured value is applied to **every** node on that
region -- not split into a single resultant -- which the Loads page's
caption states explicitly.

## Solver configuration

Both analysis types this version's GUI supports solve directly (no
iteration), so `tolerance`/`max_iterations` are validated and stored
but currently unused -- reserved for a future nonlinear GUI workflow,
with that noted directly on the Solver page.

## Results dashboard

`ResultsService.available_nodal_fields`/`available_element_fields`
report exactly what the solved result carries, so the dashboard only
ever displays a real quantity -- never a hard-coded placeholder. For a
mechanical result: max displacement, max von Mises stress, a von Mises
stress scatter plot, and a deformed-shape plot with an adjustable
display scale. For a thermal result: max/min temperature, max heat
flux, a temperature contour, and a heat-flux quiver plot -- all via the
existing Version 22 Matplotlib visualization functions.

## Version 23 3D visualization integration

The 3D Visualization page reuses `FEAViewer`
(`femtoolkit.postprocessing.visualization_3d`) directly -- it duplicates
no visualization code. Since an embedded *interactive* PyVista widget
inside a Streamlit page is out of scope for this version, the page
re-renders an off-screen `FEAViewer` screenshot to a temporary PNG on
every control change (scalar field, deformed/undeformed, deformation
scale, mesh edges) and displays it with `st.image` -- practically
"interactive" from the user's point of view, without embedding a
second rendering stack. If PyVista (the `viz3d` extra) is not
installed, the page shows a clear install message instead of failing;
the 2D Matplotlib results on the Results page remain available either
way, since Matplotlib is a core dependency.

## Running the GUI

```bash
pip install -e ".[gui,viz3d]"   # viz3d optional, for the 3D Visualization page
streamlit run src/femtoolkit/gui/app.py
```

The core FEA library and its full test suite (`pytest`) never require
Streamlit or PyVista to be installed.

## Example workflow

`examples/gui/engineering_workspace_demo/cantilever_beam_workspace.py`
runs the exact same workflow as the GUI -- Project, Material, Mesh,
Boundary Conditions, Loads, Run, Results, Visualization -- as a plain
script against `femtoolkit.application` directly, with no Streamlit
involved. Run it with:

```bash
python examples/gui/engineering_workspace_demo/cantilever_beam_workspace.py
```

It solves the toolkit's familiar 2m x 0.4m cantilever beam benchmark,
prints the maximum displacement and von Mises stress, saves stress and
deformed-shape plots, and saves the finished project configuration to
JSON.

## Troubleshooting

- **"No module named 'streamlit'"** when running `streamlit run
  src/femtoolkit/gui/app.py` -- install the `gui` extra:
  `pip install -e ".[gui]"`.
- **"3D visualization requires PyVista"** on the 3D Visualization page
  -- install the `viz3d` extra: `pip install -e ".[viz3d]"`. The 2D
  results on the Results page work without it.
- **"No boundary conditions have been defined" / other validation
  errors on the Run page** -- these are pre-solve checks
  (`femtoolkit.application.validation`); fix the named category on its
  own page (Material, Mesh, Boundary Conditions, Loads, or Solver) and
  re-run.
- **"Solver error: ..." after clicking Run Simulation** -- the model
  passed validation but the solver itself raised (e.g. a genuinely
  singular/mechanism-like configuration); the message names the
  underlying toolkit exception. This is a modeling problem, not a GUI
  bug -- check that every rigid-body mode is actually constrained.
- **A page shows "Create or load a project first"** -- every page
  except Project requires an active project; create or load one on the
  Project page first.
