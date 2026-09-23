# Engineering Simulation Reporting (Version 29)

Turns a simulation's verification results, validation results, mesh
convergence study, equilibrium checks, and reproducibility metadata into
one archivable, human-readable document. This document is the detailed
reporting guide; see [`docs/verification.md`](verification.md) and
[`docs/validation.md`](validation.md) for the underlying data this
report renders, and the main [README](../README.md#version-29) for a
shorter overview.

## Architecture: content model, then renderer

```text
VerificationReport, ValidationResult(s), MeshConvergenceStudy,
EquilibriumCheckResult(s), ReproducibilityMetadata
        |
        v
  EngineeringReport          (femtoolkit.reporting.models -- rendering-independent data)
        |
        v
  render_markdown() / render_html()   (femtoolkit.reporting.renderers)
```

`EngineeringReport` never touches Markdown, HTML, or any other output
format -- it is a plain dataclass, inspectable and testable as data.
Splitting "what the report contains" from "how it looks" means adding a
new output format is one new renderer function, not a rewrite of how
reports are assembled.

## Report sections

`EngineeringReport`'s fields mirror the eighteen sections a professional
engineering report needs:

1. Simulation Summary
2. Model Description
3. Geometry
4. Mesh
5. Materials
6. Boundary Conditions
7. Loads
8. Analysis Type
9. Solver Configuration
10. Solver Convergence
11. Verification Results
12. Validation Results
13. Mesh Convergence
14. Equilibrium Checks
15. Key Results
16. Warnings
17. Reproducibility Metadata
18. Conclusions / Status

## Building a report

```python
from femtoolkit.reporting import EngineeringReport, collect_reproducibility_metadata

metadata = collect_reproducibility_metadata(
    model_name="Cantilever Plate",
    analysis_type="linear_static",
    mesh_statistics={"nodes": 651, "elements": 600, "dofs": 1302},
    element_types=["QuadElement2D"],
    material_properties={"youngs_modulus": 200e9, "poisson_ratio": 0.3},
    boundary_conditions_summary="Fixed (X, Y) on the left edge.",
    loads_summary="10000 N downward at the tip.",
    degrees_of_freedom=1302,
)

report = EngineeringReport(
    title="Cantilever Plate: Engineering Simulation Report",
    simulation_summary="...", model_description="...", geometry_description="...",
    mesh_summary={...}, materials_summary={...},
    boundary_conditions_summary="...", loads_summary="...",
    analysis_type="linear_static", solver_configuration="...",
    reproducibility=metadata,
    verification_results=benchmark_report.results,   # from VerificationRunner
    equilibrium_checks=[equilibrium_check],           # from check_force_equilibrium
    key_results={"Maximum displacement (m)": 0.0021},
    conclusions="All benchmark cases passed; the model satisfies global equilibrium.",
)
```

## No unsupported engineering conclusions

**`conclusions` is free text the caller supplies. The framework never
generates an engineering judgment automatically** -- there is no code
path anywhere in `femtoolkit.reporting` that writes a sentence like
"the structure is safe" or "the design is adequate." The only thing
computed automatically is `EngineeringReport.overall_status`, a purely
factual tally of the structured statuses already present in
`verification_results`, `validation_results`, `equilibrium_checks`, and
`mesh_convergence` (`FAIL` if anything failed, else `WARNING` if
anything needs review, else `NOT_AVAILABLE` if anything could not be
evaluated, else `PASS`, or `NOT_RUN` if there is nothing to report on
at all) -- a status label, not a claim about real-world fitness for
purpose.

## Reproducibility metadata

`femtoolkit.reporting.metadata.ReproducibilityMetadata` records
everything spec section 11 asks for: toolkit version, Python version,
platform, dependency versions (`numpy`/`scipy`/`matplotlib`, looked up
via `importlib.metadata` and simply omitted -- not errored -- if a
lookup fails), model name, analysis type, mesh statistics, element
types, material properties, boundary-condition/load summaries, solver
name, tolerances, degrees of freedom, execution mode (Version 27), and
an optional random seed. `collect_reproducibility_metadata(...)`
builds one for the current run; `ReproducibilityMetadata.to_dict()`
returns a JSON-serializable representation.

## Rendering and export

```python
from femtoolkit.reporting import render_markdown, render_html, save_report

markdown_text = render_markdown(report)
html_text = render_html(report)
save_report(report, "report.md", "markdown")
save_report(report, "report.html", "html")
```

Both renderers are plain string formatting -- no templating engine, no
new dependency. Markdown is the primary format (diff-friendly, suitable
for archiving in version control alongside a project's history); HTML
wraps the same content, escaped, for direct viewing in a browser.

### Why no PDF

PDF export was evaluated and deliberately **not added**. Producing a
well-formatted PDF needs either a heavy rendering dependency (a headless
browser engine, a LaTeX toolchain) or a hand-rolled layout engine --
neither justified for a document format Markdown and HTML already serve
well, and both would violate this toolkit's consistent "no unnecessary
dependencies" policy (the same reasoning that kept Version 27's
parallel execution on the standard library alone). A user who needs a
PDF can convert the Markdown/HTML output with a tool of their choosing
outside this toolkit.

## GUI integration

The GUI's Verification & Validation page
(`femtoolkit.gui.workflow_pages.verification_page`) assembles an
`EngineeringReport` from the current project's solved run, the last-run
benchmark suite results, and the equilibrium check
`SimulationService` already computed, then offers both formats as
`st.download_button` downloads with a live Markdown preview -- using
exactly this module's public API, with no reporting logic duplicated in
the GUI layer.

## Limitations

No PDF renderer (see above). No chart/figure embedding in the rendered
report itself (verification plots are separate `matplotlib.figure.Figure`
objects, saved independently -- see
[`docs/verification.md`](verification.md#verification-plots)). No
report versioning/diffing tools beyond what version control provides
for the saved Markdown file. No multi-run comparison report (planned as
part of the Version 30 preview's "result comparison dashboards").
