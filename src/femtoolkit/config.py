"""Minimal package configuration for the Finite Element Toolkit.

Configuration is kept to a bare minimum: package metadata and the default
logging level. Advanced solver settings (convergence tolerances, iteration
limits, sparse-solver backends, etc.) belong to future versions once the
toolkit supports more than the basic dense linear solve introduced in
Version 2.
"""

from __future__ import annotations

import logging

__version__: str = "3.0.0"
"""Current version of the Finite Element Toolkit package."""

PACKAGE_NAME: str = "femtoolkit"
"""Distribution and top-level import name of the package."""

DEFAULT_LOG_LEVEL: int = logging.WARNING
"""Default logging level applied to the package logger.

``WARNING`` is used so that importing the toolkit does not produce noisy
output in applications that embed it, while still surfacing genuine
problems by default.
"""
