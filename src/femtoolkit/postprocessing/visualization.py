"""Turning a :class:`~femtoolkit.postprocessing.result_model.SimulationResult` into plots.

Uses `Matplotlib <https://matplotlib.org/>`_, the standard, minimal,
non-interactive plotting library for engineering/scientific Python --
introduced as a project dependency specifically for this version (see
the Version 22 section of the project README for why: no prior version
needed one, and this toolkit's core solvers remain completely
independent of it, since only this module imports it). Every function
here returns a :class:`matplotlib.figure.Figure` and never calls
``plt.show()`` -- building the figure is this module's job; whether to
display it, save it to a file, or embed it elsewhere is the caller's
(see the ``examples/post_processing`` scripts, which save every figure
to a PNG file rather than opening an interactive window, per this
version's explicit "no GUI" scope).

**Scope: 2D plots.** Section 12 of this version's brief asks for 1D
plots and 2D (filled) contour plots, with 3D deferred to "where existing
visualization tools allow" -- this toolkit has none yet (see the
Version 23 preview, PyVista-based). Every function here therefore reads
only the ``x``/``y`` coordinates of a result's topology; a 3D mesh's
``z`` coordinate is ignored (documented per-function), appropriate for
this version's scope and consistent with not building a 3D viewer here.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from femtoolkit.exceptions import ValidationError
from femtoolkit.postprocessing.field_calculator import deformed_coordinates, vector_magnitude
from femtoolkit.postprocessing.result_model import SimulationResult


def _field_label(result: SimulationResult, field_name: str, component: int | None) -> str:
    unit = result.field_units.get(field_name, "")
    label = field_name.replace("_", " ")
    if component is not None:
        label = f"{label} [{component}]"
    return f"{label} ({unit})" if unit else label


def plot_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Figure:
    """Create a simple 1D line plot (a temperature profile, a time history, ...).

    Args:
        x_values: The horizontal axis values.
        y_values: The vertical axis values, same length as ``x_values``.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        title: Plot title.

    Returns:
        A :class:`matplotlib.figure.Figure` with one line plot, gridlines,
        and axis labels.
    """
    figure = Figure(figsize=(7.0, 5.0))
    axes = figure.add_subplot(111)
    axes.plot(x_values, y_values, marker="o", markersize=4, linewidth=1.5)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_semilog_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Figure:
    """Create a 1D line plot with a logarithmic vertical axis (a residual history, ...).

    Version 29 addition, alongside :func:`plot_line`: a quantity like a
    solver's residual norm typically spans several orders of magnitude
    over the course of a solve, where a linear vertical axis would
    compress every iteration but the first few into an indistinguishable
    flat line near zero -- the standard visualization is a semi-log plot.

    Args:
        x_values: The horizontal axis values (e.g. iteration number).
        y_values: The vertical axis values, same length as ``x_values``.
            Must be strictly positive (a logarithmic axis cannot
            represent zero or negative values).
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        title: Plot title.

    Returns:
        A :class:`matplotlib.figure.Figure` with one semi-log line plot.
    """
    figure = Figure(figsize=(7.0, 5.0))
    axes = figure.add_subplot(111)
    axes.semilogy(x_values, y_values, marker="o", markersize=4, linewidth=1.5)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(visible=True, which="both", alpha=0.3)
    figure.tight_layout()
    return figure


def plot_comparison_line(
    x_values: np.ndarray,
    numerical_values: np.ndarray,
    reference_values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    numerical_label: str = "FEA result",
    reference_label: str = "Reference",
) -> Figure:
    """Create a 1D line plot overlaying a numerical result against a reference.

    Version 29 addition, alongside :func:`plot_line`: the standard way
    to visually compare a verification or validation result against its
    analytical/reference counterpart -- two series on one set of axes,
    distinguished by marker style rather than color alone (so the plot
    stays readable in grayscale or for a color-blind reader).

    Args:
        x_values: The horizontal axis values, shared by both series.
        numerical_values: The FEA (numerical) series.
        reference_values: The reference series, same length as
            ``numerical_values``.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        title: Plot title.
        numerical_label: Legend label for ``numerical_values``.
        reference_label: Legend label for ``reference_values``.

    Returns:
        A :class:`matplotlib.figure.Figure` with both series plotted and
        a legend.
    """
    figure = Figure(figsize=(7.0, 5.0))
    axes = figure.add_subplot(111)
    axes.plot(
        x_values, reference_values, marker="s", markersize=5, linewidth=1.5, label=reference_label
    )
    axes.plot(
        x_values,
        numerical_values,
        marker="o",
        markersize=4,
        linewidth=1.0,
        linestyle="--",
        label=numerical_label,
    )
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(visible=True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure


def plot_nodal_contour_2d(
    result: SimulationResult,
    field_name: str,
    step: int = -1,
    component: int | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Create a filled 2D contour plot of a nodal scalar field.

    Uses the node ``(x, y)`` coordinates directly (``z`` is ignored --
    see the module docstring) with :meth:`matplotlib.axes.Axes.tricontourf`,
    which triangulates the (generally unstructured) node positions
    itself -- no structured grid is required, so this works directly for
    a CST or Q4 mesh of any shape.

    Args:
        result: The result to plot.
        field_name: The nodal field to contour (e.g. ``"temperature"``).
        step: Which step to plot (default: the last).
        component: Which vector component to plot, or ``None`` for a
            scalar field.
        cmap: Matplotlib colormap name.

    Returns:
        A :class:`matplotlib.figure.Figure` with a filled contour plot,
        colorbar, and axis labels.
    """
    result_step = result.step(step)
    values = result_step.nodal_field(field_name)
    node_ids = list(values)
    x = np.array([result.topology.node_coordinates[node_id][0] for node_id in node_ids])
    y = np.array([result.topology.node_coordinates[node_id][1] for node_id in node_ids])
    if component is not None:
        z = np.array([np.asarray(values[node_id])[component] for node_id in node_ids])
    else:
        z = np.array([float(np.asarray(values[node_id])) for node_id in node_ids])

    figure = Figure(figsize=(7.0, 6.0))
    axes = figure.add_subplot(111)
    contour = axes.tricontourf(x, y, z, levels=20, cmap=cmap)
    figure.colorbar(contour, ax=axes, label=_field_label(result, field_name, component))
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(f"{field_name.replace('_', ' ').title()} (step {result_step.index})")
    figure.tight_layout()
    return figure


def plot_element_scatter_2d(
    result: SimulationResult,
    field_name: str,
    step: int = -1,
    component: int | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Create a colored scatter plot of an element scalar field, at each element's centroid.

    Args:
        result: The result to plot.
        field_name: The element field to plot (e.g. ``"von_mises_stress"``).
        step: Which step to plot (default: the last).
        component: Which vector component to plot, or ``None`` for a
            scalar field.
        cmap: Matplotlib colormap name.

    Returns:
        A :class:`matplotlib.figure.Figure` with a colored scatter plot,
        colorbar, and axis labels.
    """
    result_step = result.step(step)
    values = result_step.element_field(field_name)
    element_ids = list(values)

    centroids_x = []
    centroids_y = []
    for element_id in element_ids:
        node_ids = result.topology.element_connectivity[element_id]
        coords = np.array([result.topology.node_coordinates[node_id] for node_id in node_ids])
        centroids_x.append(coords[:, 0].mean())
        centroids_y.append(coords[:, 1].mean())

    if component is not None:
        z = np.array([np.asarray(values[element_id])[component] for element_id in element_ids])
    else:
        z = np.array([float(np.asarray(values[element_id])) for element_id in element_ids])

    figure = Figure(figsize=(7.0, 6.0))
    axes = figure.add_subplot(111)
    scatter = axes.scatter(centroids_x, centroids_y, c=z, cmap=cmap, s=120, edgecolors="black")
    figure.colorbar(scatter, ax=axes, label=_field_label(result, field_name, component))
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_aspect("equal", adjustable="box")
    axes.set_title(f"{field_name.replace('_', ' ').title()} (step {result_step.index})")
    figure.tight_layout()
    return figure


def plot_heat_flux_vectors_2d(result: SimulationResult, step: int = -1) -> Figure:
    """Create a quiver (vector arrow) plot of heat flux, one arrow per element centroid.

    Args:
        result: The result to plot (must have a ``"heat_flux"`` element field).
        step: Which step to plot (default: the last).

    Returns:
        A :class:`matplotlib.figure.Figure` with a quiver plot.

    Raises:
        ValidationError: If ``result`` has no ``"heat_flux"`` element field.
    """
    result_step = result.step(step)
    flux_field = result_step.element_field("heat_flux")

    centroids_x, centroids_y, flux_x, flux_y = [], [], [], []
    for element_id, flux in flux_field.items():
        node_ids = result.topology.element_connectivity[element_id]
        coords = np.array([result.topology.node_coordinates[node_id] for node_id in node_ids])
        centroids_x.append(coords[:, 0].mean())
        centroids_y.append(coords[:, 1].mean())
        vector = np.asarray(flux)
        flux_x.append(vector[0])
        flux_y.append(vector[1] if vector.size > 1 else 0.0)

    figure = Figure(figsize=(7.0, 6.0))
    axes = figure.add_subplot(111)
    axes.quiver(centroids_x, centroids_y, flux_x, flux_y, color="firebrick")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_aspect("equal", adjustable="box")
    unit = result.field_units.get("heat_flux", "")
    axes.set_title(f"Heat Flux Vectors ({unit}, step {result_step.index})")
    figure.tight_layout()
    return figure


def plot_deformed_shape_2d(
    result: SimulationResult,
    scale: float = 1.0,
    step: int = -1,
    color_field: str | None = None,
) -> Figure:
    """Plot a 2D mesh's original and deformed (scaled) outlines.

    Each element's own node order is used to draw its outline as a
    closed polygon -- appropriate for a 2D CST/Q4 mesh (see the module
    docstring for why this is 2D-only).

    Args:
        result: The result to plot (must have a ``"displacement"`` nodal
            field).
        scale: The visualization scale factor applied to the
            displacement (see
            :func:`~femtoolkit.postprocessing.field_calculator.deformed_coordinates`).
            Purely cosmetic.
        step: Which step to plot (default: the last).
        color_field: An optional element scalar field to color the
            deformed elements by (e.g. ``"von_mises_stress"``). If
            ``None``, the deformed shape is drawn as an unfilled outline.

    Returns:
        A :class:`matplotlib.figure.Figure` showing the original
        (dashed, light) and deformed (solid) mesh outlines.
    """
    result_step = result.step(step)
    displacement_field = result_step.nodal_field("displacement")
    deformed = deformed_coordinates(result.topology, displacement_field, scale)

    figure = Figure(figsize=(7.0, 6.0))
    axes = figure.add_subplot(111)

    for node_ids in result.topology.element_connectivity.values():
        original = np.array([result.topology.node_coordinates[n] for n in node_ids])
        axes.fill(
            original[:, 0], original[:, 1], edgecolor="gray", facecolor="none",
            linestyle="--", linewidth=1.0,
        )

    if color_field is not None:
        values = result_step.element_field(color_field)
        polygons = []
        colors = []
        for element_id, node_ids in result.topology.element_connectivity.items():
            if element_id not in values:
                continue
            polygons.append(np.array([deformed[n] for n in node_ids])[:, :2])
            colors.append(float(np.asarray(values[element_id])))
        from matplotlib.collections import PolyCollection

        collection = PolyCollection(polygons, array=np.array(colors), cmap="viridis")
        axes.add_collection(collection)
        figure.colorbar(collection, ax=axes, label=_field_label(result, color_field, None))
    else:
        for node_ids in result.topology.element_connectivity.values():
            deformed_polygon = np.array([deformed[n] for n in node_ids])
            axes.fill(
                deformed_polygon[:, 0], deformed_polygon[:, 1],
                edgecolor="firebrick", facecolor="none", linewidth=1.5,
            )

    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_aspect("equal", adjustable="box")
    axes.autoscale_view()
    axes.set_title(f"Deformed Shape (scale={scale:g}, step {result_step.index})")
    figure.tight_layout()
    return figure


def plot_time_history(
    result: SimulationResult,
    field_name: str,
    node_id: int | None = None,
    element_id: int | None = None,
    component: int | None = None,
) -> Figure:
    """Plot one entity's value for one field across every step (time or load history).

    Args:
        result: The result to plot.
        field_name: The field to plot.
        node_id: The node to plot a nodal field's history for. Exactly
            one of ``node_id``/``element_id`` must be given.
        element_id: The element to plot an element field's history for.
        component: Which vector component to plot, or ``None`` for a
            scalar field (or a vector field's magnitude -- see
            :func:`~femtoolkit.postprocessing.field_calculator.vector_magnitude`).

    Returns:
        A :class:`matplotlib.figure.Figure` with a time/load-step history line plot.

    Raises:
        ValidationError: If both or neither of ``node_id``/``element_id`` are given.
    """
    if (node_id is None) == (element_id is None):
        raise ValidationError("plot_time_history requires exactly one of node_id or element_id.")

    if node_id is not None:
        if component is None:
            history = np.array(
                [
                    vector_magnitude(step.nodal_value(field_name, node_id))
                    for step in result.steps
                ]
            )
        else:
            history = result.nodal_history(field_name, node_id, component)
        entity_label = f"node {node_id}"
    else:
        if component is None:
            history = np.array(
                [
                    vector_magnitude(step.element_value(field_name, element_id))
                    for step in result.steps
                ]
            )
        else:
            history = result.element_history(field_name, element_id, component)
        entity_label = f"element {element_id}"

    return plot_line(
        result.times,
        history,
        xlabel="Time / Load Factor",
        ylabel=_field_label(result, field_name, component),
        title=f"{field_name.replace('_', ' ').title()} History at {entity_label}",
    )
