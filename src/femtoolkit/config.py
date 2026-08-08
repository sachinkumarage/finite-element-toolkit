"""Minimal package configuration for the Finite Element Toolkit.

Version 1 keeps configuration to a bare minimum: package metadata and the
default logging level. Numerical solver settings (tolerances, iteration
limits, etc.) belong to future versions once a solver actually exists.
"""

from __future__ import annotations

import logging

__version__: str = "1.0.0"
"""Current version of the Finite Element Toolkit package."""

PACKAGE_NAME: str = "femtoolkit"
"""Distribution and top-level import name of the package."""

DEFAULT_LOG_LEVEL: int = logging.WARNING
"""Default logging level applied to the package logger.

``WARNING`` is used so that importing the toolkit does not produce noisy
output in applications that embed it, while still surfacing genuine
problems by default.
"""
