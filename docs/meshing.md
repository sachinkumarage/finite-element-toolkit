# Advanced Meshing & Engineering Model Preparation (Version 25)

An engineering-oriented mesh-preparation workflow layered on top of the
existing mesh domain model: `Geometry -> Mesh Generation -> Mesh
Validation -> Mesh Quality Assessment -> Mesh Refinement -> Mesh
Visualization -> FEA Solver`. This document is the detailed technical
guide; see the main [README](../README.md#version-25) for a shorter
overview and how Version 25 fits into the toolkit's history.

## Why mesh quality matters

A finite element solution's accuracy depends strongly on element
shape, not just element count. Poorly shaped elements can cause:

- **Numerical instability** -- a nearly-singular strain-displacement
  matrix amplifies round-off error.
- **Inaccurate stress/strain results** -- a distorted element's
  interpolated displacement field poorly approximates the true
  solution.
- **Poor convergence** -- for nonlinear analyses, a badly conditioned
  element stiffness slows or prevents Newton-Raphson convergence.
- **Distorted deformation** -- a nearly-degenerate element can produce
  visually implausible deformed shapes.
- **Ill-conditioned system matrices** -- extreme aspect ratios or
  near-zero Jacobian determinants push the global stiffness matrix's
  condition number higher, degrading the solver's numerical accuracy.

A good mesh represents the geometry accurately while keeping every
element's shape close to its "ideal" form (a square for Q4/HEX8, an
equilateral triangle for CST, a regular tetrahedron for TET4).

## Architecture

```text
femtoolkit.mesh.quality/        Element and mesh shape-quality metrics
femtoolkit.mesh.validation/     Structural/geometric validity checking
femtoolkit.mesh.statistics      Descriptive mesh statistics
femtoolkit.mesh.sizing          Mesh sizing abstraction (target/min/max size)
femtoolkit.mesh.generation/     Pluggable mesh-generator architecture
femtoolkit.mesh.refinement      Uniform/local mesh refinement
```

`femtoolkit.mesh.quality` and `femtoolkit.mesh.validation` were single
modules through Version 24 (`quality.py`, `validation.py`); both are
reorganized into packages here **without changing any existing public
name or import path** -- `from femtoolkit.mesh.quality import
ElementQuality, MeshQualitySummary, compute_element_quality,
compute_mesh_quality_summary` and `from femtoolkit.mesh.validation
import validate_mesh` keep working exactly as they did in every prior
version. Nothing in this version modifies `femtoolkit.mesh.mesh`,
`femtoolkit.mesh.generator`, or any element class -- the new
subsystems are purely additive.

## Mesh quality metrics

`femtoolkit.mesh.quality.metrics` computes, for each supported element
type, only the metrics that are mathematically meaningful for its own
formulation:

### 2D continuum elements (CST, Q4) -- unchanged since Version 8

- **Area**, **min/max edge length**.
- **Aspect ratio** -- `max_edge_length / min_edge_length` (>= 1.0).
- **Quality** -- `min_edge_length / max_edge_length`, in `(0, 1]`
  (1.0 = best possible shape).
- **Skewness** -- the equiangle skew, how far the most-distorted
  interior angle is from the ideal angle (60 degrees for CST, 90 for
  Q4), normalized to `[0, 1]`.
- **Jacobian determinant** -- at the element center, for Q4 only
  (`None` for CST, which has no isoparametric mapping in the same
  sense).

### 3D solid elements (TET4, HEX8) -- new in Version 25

- **Volume**, **min/max edge length** (6 edges for TET4, 12 for HEX8),
  **aspect ratio**, **quality** (same edge-ratio definitions as above).
- **Characteristic size** -- `volume ** (1/3)`.
- **Jacobian determinant**:
  - **TET4** has linear shape functions, so its Jacobian is *constant*
    over the element -- one value, not a minimum. TET4 also follows a
    documented "either node winding is valid" orientation policy (see
    `femtoolkit.mesh.tet4_element`'s module docstring), so a negative
    determinant does **not**, by itself, indicate an invalid element
    for TET4 -- only a near-zero one does (already guarded by
    `MIN_TETRAHEDRON_VOLUME` at construction). The reported value
    keeps its sign for transparency.
  - **HEX8** is isoparametric with trilinear shape functions, so the
    Jacobian varies across the element; the metric reported is the
    **minimum** determinant over the same 2x2x2 Gauss points used by
    the element's own stiffness integration
    (`femtoolkit.continuum.gauss.GAUSS_2X2X2_POINTS`). HEX8
    construction already rejects a non-positive Jacobian at any Gauss
    point (see `femtoolkit.continuum.jacobian`), so for any
    successfully constructed `Hex8Element3D` this value is always
    positive -- a **distortion measure**, not a validity gate (which
    is already enforced at construction time).

$$
\det(\mathbf{J}) = \det\left(\frac{\partial(x,y,z)}{\partial(\xi,\eta,\zeta)}\right)
$$

`compute_any_element_quality(element)` dispatches to the 2D or 3D
function by element type and raises `UnsupportedQualityMetricError` for
an element type with no defined quality concept (bar, truss, frame --
a 1D line has no isoparametric area/volume mapping).

## The quality evaluator and report

```python
from femtoolkit.mesh.quality import QualityEvaluator

evaluator = QualityEvaluator(poor_quality_threshold=0.3)
report = evaluator.evaluate(mesh)

report.num_elements_evaluated
report.minimum_quality
report.maximum_quality
report.mean_quality
report.poor_quality_element_ids   # elements below the threshold
report.warnings                   # e.g. "3 element(s) have poor quality..."
```

`QualityEvaluator.evaluate` works across **any mix** of CST/Q4/TET4/HEX8
elements in one mesh (bar/truss/frame elements are silently skipped,
not treated as an error), returning a typed
`femtoolkit.mesh.quality.quality_report.MeshQualityReport` -- distinct
from the unchanged, 2D-only `MeshQualitySummary`
(`compute_mesh_quality_summary`) every Version 8-24 caller already
uses.

## Mesh validation

`femtoolkit.mesh.validation.validate_mesh` (unchanged since Version 8)
is a **fail-fast** check: it raises on the first problem found (a
duplicate node coordinate, an invalid element reference, a
non-positive area), appropriate for a "should the solver trust this
mesh" gate -- `femtoolkit.mesh.generator` calls it on its own output
before returning.

`femtoolkit.mesh.validation.generate_validation_report` is a new,
complementary, **non-raising** sweep that collects every problem it can
find and returns one structured report:

```python
from femtoolkit.mesh.validation import generate_validation_report, format_report

report = generate_validation_report(mesh)
print(format_report(report))
```

```text
Mesh Validation Report
----------------------------

Status: OK

Nodes:
  Total: 1250
  Duplicate: 0
  Isolated: 0

Elements:
  Total: 980
  Invalid connectivity: 0
  Degenerate: 0
  Duplicate: 0

Quality:
  Minimum: 0.71
  Mean: 0.94
```

Checked, without raising:

- **Topology** -- invalid element connectivity (a node reference
  absent from the mesh).
- **Geometry** -- degenerate elements (non-positive area/volume).
- **Duplicates** -- duplicate node coordinates, duplicate elements
  (same type, same node set).
- **Connectivity** -- isolated nodes (referenced by no element).
- **Quality** -- an embedded `MeshQualityReport`, with its warnings
  folded into the validation report's own warnings.

`status` is `"OK"` (nothing found), `"WARNING"` (usable but with
quality or non-fatal structural concerns), or `"ERROR"` (a structural
problem -- duplicate nodes or invalid connectivity -- that should be
fixed). Most individual checks can never actually fail for a mesh built
through this toolkit's normal APIs (`Mesh` already guards against
invalid references at construction time, and every element type
validates its own geometry in `__post_init__`); they remain valuable
for a mesh reconstructed from external data or a future,
less-constrained generator. Nothing is ever automatically deleted or
"fixed" -- the report only reports.

## Mesh refinement

`femtoolkit.mesh.refinement.refine_uniform(mesh, elements=None)` only
refines `CSTElement2D` and `QuadElement2D` -- the two element types
with a mathematically reliable uniform subdivision rule. TET4/HEX8
subdivision templates are geometrically more involved (a tetrahedron's
central-octahedron split has several valid diagonal choices that
affect resulting quality; a hexahedron's 1-to-8 split needs face- and
volume-center nodes in addition to edge midpoints), so they are
deliberately left for a future version rather than shipped as an
unreliable approximation.

**Uniform refinement** (`elements=None`, the default) applies "1-to-4"
edge-midpoint quadrisection to every CST/Q4 element:

$$
T \rightarrow 4T \qquad\qquad Q \rightarrow 4Q
$$

**Local (selective) refinement** (`elements={id, ...}`) refines only
the named elements, leaving the rest of the mesh untouched. This can
produce a **non-conforming mesh** (a "hanging node": a midpoint node on
a shared edge that only one side's element actually uses) -- a known,
documented limitation, not a defect. A fully conformal local/adaptive
refinement scheme is future-version scope (see the Version 26 preview;
spec section 8 for this version explicitly excludes adaptive
error-based refinement).

**Shared edges never get a duplicate midpoint node**: refining two
neighboring elements that share an edge reuses the same midpoint node
for both, keyed by the edge's node-ID pair. Every refined mesh is
validated automatically (`validate_mesh`) before being returned, and
refined elements inherit their parent's material and thickness
unchanged. Total area is conserved to floating-point precision, and
boundary node locations are preserved exactly (an original boundary
node stays on the boundary; a new midpoint on a boundary edge lies
exactly on that boundary too), so region-based boundary-condition
selection (`Mesh.nodes_on_boundary`) keeps working correctly after
refinement.

## Mesh sizing

`femtoolkit.mesh.sizing.MeshSizingParameters` expresses "how fine
should the mesh be" in physical units rather than raw subdivision
counts:

```python
from femtoolkit.mesh.sizing import MeshSizingParameters

sizing = MeshSizingParameters(target_size=0.1, minimum_size=0.02, maximum_size=0.5)
nx, ny = sizing.subdivisions(width=2.0, height=0.4)  # (20, 4)
```

**Units.** Every length in this toolkit -- node coordinates, domain
width/height, element thickness -- is a plain `float` interpreted as
**meters** (SI base unit); nothing in the codebase attaches a unit tag
to a number. `target_size`/`minimum_size`/`maximum_size` follow the
same convention. `femtoolkit.units` provides the toolkit's existing SI
unit constants for a caller converting from another unit system before
constructing a `MeshSizingParameters` -- there is no second unit
system here.

`regional_sizes` (mapping a named boundary region to a target size) is
accepted and validated but **not yet consumed** by this version's
structured generators, which produce a single uniform grid -- an
honest interface for a future graded/unstructured generator, not a
fabricated capability.

## Mesh generator architecture

`femtoolkit.mesh.generation.base.MeshGenerator` is a `typing.Protocol`
(structural typing, not an abstract base class) that a future
unstructured, tetrahedral, or hexahedral generator can satisfy without
inheriting from anything -- neither `femtoolkit.mesh.mesh` nor the
solver ever need to change to accommodate a new generator:

```python
from femtoolkit.mesh.generation import StructuredQuadMeshGenerator
from femtoolkit.mesh.sizing import MeshSizingParameters
from femtoolkit.geometry import Rectangle

generator = StructuredQuadMeshGenerator()
mesh = generator.generate(
    Rectangle(width=2.0, height=0.4), MeshSizingParameters(target_size=0.1), material, thickness=0.02
)
```

`StructuredQuadMeshGenerator`/`StructuredTriangularMeshGenerator` are
thin adapters: all the actual node/element generation logic remains
exactly where it was in Version 8
(`femtoolkit.mesh.generator.create_quad_mesh`/`create_triangular_mesh`),
unchanged and unduplicated.

## Boundary preservation

Region-based node selection (`Mesh.nodes_on_boundary`, Version 9) works
unchanged on a refined mesh, since refinement preserves every original
node's location exactly and places new edge-midpoint nodes exactly on
the edge they subdivide (including boundary edges). No new region or
material-association concept was needed: refined elements simply keep
their parent's `material`/`thickness` attributes directly.

## Mesh quality visualization

The 3D quality visualization reuses Version 23's PyVista mesh-conversion
layer directly (`femtoolkit.postprocessing.visualization_3d.mesh_converter`)
rather than building a second visualization framework:

```text
Mesh -> Quality Evaluation -> Scalar Quality Field -> PyVista Grid -> Screenshot
```

```python
from femtoolkit.mesh.quality import QualityEvaluator
from femtoolkit.gui.visualization import render_mesh_quality_screenshot

report = QualityEvaluator().evaluate(mesh)
render_mesh_quality_screenshot(mesh, report, "quality.png", metric="quality")
```

`metric` is one of `"quality"`, `"aspect_ratio"`, or
`"jacobian_determinant"`. Since `quality` is defined to lie in `(0, 1]`
by construction, its color range is fixed to `[0, 1]` rather than
auto-scaled to the data's min/max -- auto-scaling on a near-perfect
mesh would otherwise turn sub-1e-10 floating-point noise into a
misleading full red-to-green gradient. This requires the `viz3d` extra
(`pip install "femtoolkit[viz3d]"`); it is not available otherwise, and
callers should check `femtoolkit.gui.visualization.is_pyvista_available()`
first.

## GUI workflow

The Mesh page in the Version 24 Engineering Workspace
(`streamlit run src/femtoolkit/gui/app.py`) implements the full Model
Preparation workflow:

```text
1. Configure Mesh   (geometry + sizing: global/min/max target size)
2. Generate Mesh
3. Validate Mesh     (structured report + status)
4. Evaluate Quality  (dashboard: elements evaluated, min/mean quality, poor elements)
5. Mesh Statistics   (node/element counts, type distribution, bounding box, characteristic size)
6. Refine Mesh       (uniform, applies at the next generation -- including at solve time)
7. Inspect Mesh      (2D Matplotlib preview + 3D PyVista quality preview)
8. Accept Mesh       (no separate step: the configuration already feeds the solver)
```

Every button calls `femtoolkit.application.mesh_preparation_service.MeshPreparationService`
-- the GUI page itself performs no meshing algorithm. Refinement is
wired all the way through to a real simulation run: `Project.mesh.refinement_passes`
(new in this version, default `0`, fully backward-compatible with every
Version 24 saved project) is applied by
`femtoolkit.application.model_service.ModelService.build_mesh` via
`refine_uniform` before the mesh reaches `StaticLinearAnalysis`/
`SteadyStateThermalAnalysis` -- so "Refine" on the Mesh page genuinely
changes what gets solved, with no separate "accept" action needed.

## Examples

```bash
python examples/meshing/mesh_quality_demo.py       # quality evaluation, good vs. distorted mesh
python examples/meshing/mesh_validation_demo.py    # structured validation reports (OK/WARNING)
python examples/meshing/mesh_refinement_demo.py    # uniform + local refinement, area conservation
```

## Limitations

- Refinement supports only CST/Q4 (uniform "1-to-4" quadrisection);
  TET4/HEX8 have no refinement rule yet.
- Local refinement can produce a non-conforming (hanging-node) mesh --
  a documented, accepted limitation, not a bug; fully conformal
  adaptive refinement is Version 26+ scope.
- Mesh generation remains **structured** only (a single uniform
  rectangular grid); no unstructured, CAD-driven, or automatic mesh
  generation.
- `MeshSizingParameters.regional_sizes` is validated but not yet
  consumed by the structured generators (no per-region element size
  variation yet).
- No geometry-to-mesh workflow beyond the existing
  `femtoolkit.geometry.Rectangle`; no CAD kernel or CAD file import.
- Mesh-quality visualization requires the optional `viz3d` (PyVista)
  extra; the 2D Matplotlib mesh preview remains available without it.

## Version 26 preview (not implemented)

**Advanced Solver Infrastructure & Sparse Linear Algebra** -- sparse
global matrices, sparse linear solvers, solver backends,
preconditioning, iterative solvers, solver performance monitoring,
memory-efficient assembly, and solver diagnostics, aimed at handling
substantially larger engineering models efficiently than the current
dense-matrix `numpy.linalg.solve` path.
