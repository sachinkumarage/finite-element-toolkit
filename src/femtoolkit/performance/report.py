"""Human-readable formatting for :class:`~femtoolkit.performance.profiler.PerformanceReport`.

Version 27. Kept separate from :mod:`femtoolkit.performance.profiler` so that
measuring performance never depends on how (or whether) it is ever
printed -- the same separation :mod:`femtoolkit.postprocessing.visualization`
(Version 22) keeps between computing results and displaying them.
"""

from __future__ import annotations

from femtoolkit.performance.profiler import PerformanceReport

_LABELS: tuple[tuple[str, str], ...] = (
    ("mesh_time", "Mesh preparation"),
    ("element_time", "Element calculations"),
    ("assembly_time", "Assembly"),
    ("boundary_condition_time", "Boundary conditions"),
    ("solve_time", "Solve"),
    ("post_processing_time", "Post-processing"),
)


def format_report(report: PerformanceReport) -> str:
    """Render a :class:`PerformanceReport` as a labeled, human-readable block.

    Only fields that were actually measured are shown -- a stage that was
    never profiled (``None``) is omitted rather than printed as ``0.00
    s``, which would misleadingly imply it took no time (spec section 4/20:
    never report a metric that was not measured).

    Args:
        report: The report to format.

    Returns:
        A multi-line string, e.g.::

            Performance Report
            ------------------------------
            Model: Cantilever Beam
            Nodes: 5200
            Elements: 4800
            Execution: parallel (4 workers)

            Element calculations: 0.4200 s
            Assembly: 0.1800 s
            Solve: 0.3100 s
            Post-processing: 0.1400 s

            Total: 1.0500 s
    """
    lines = ["Performance Report", "-" * 30]

    if report.model_name is not None:
        lines.append(f"Model: {report.model_name}")
    if report.node_count is not None:
        lines.append(f"Nodes: {report.node_count}")
    if report.element_count is not None:
        lines.append(f"Elements: {report.element_count}")
    if report.execution_mode is not None:
        workers_suffix = f" ({report.workers} workers)" if report.workers is not None else ""
        lines.append(f"Execution: {report.execution_mode}{workers_suffix}")
    if report.solver_result is not None:
        lines.append(f"Solver: {report.solver_result.solver_name}")
        dofs = report.solver_result.diagnostics.get("dofs")
        if dofs is not None:
            lines.append(f"DOFs: {dofs}")
        nnz = report.solver_result.diagnostics.get("nnz")
        if nnz is not None:
            lines.append(f"Non-zero entries: {nnz}")

    lines.append("")
    for field_name, label in _LABELS:
        value = getattr(report, field_name)
        if value is not None:
            lines.append(f"{label}: {value:.4f} s")

    lines.append("")
    lines.append(f"Total: {report.total_time:.4f} s")
    return "\n".join(lines)


__all__ = ["format_report"]
