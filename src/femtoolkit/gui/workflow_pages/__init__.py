"""One module per GUI workflow step: Project, Material, Mesh, Boundary
Conditions, Loads, Solver, Run, Results, Visualization (Version 24).

Each page module exposes a single ``render(state: AppState) -> None``
function; :mod:`femtoolkit.gui.app` is the only place that decides which
page to render, keeping every page a plain, independently readable
function rather than a class hierarchy.
"""
