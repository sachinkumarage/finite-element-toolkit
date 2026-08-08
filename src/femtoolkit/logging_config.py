"""Package-level logging configuration for the Finite Element Toolkit.

Libraries should not configure the root logger or otherwise change how a
host application logs. This module follows the standard library's
recommended pattern for library logging: it obtains a single named logger
for the package and attaches a :class:`logging.NullHandler` to it, so that
the toolkit produces no output at all unless the embedding application
explicitly configures handlers for it.
"""

from __future__ import annotations

import logging

from femtoolkit.config import DEFAULT_LOG_LEVEL, PACKAGE_NAME


def get_logger(name: str = PACKAGE_NAME) -> logging.Logger:
    """Return the package logger, or a child logger under it.

    Args:
        name: Logger name. Defaults to the package name. Pass a
            module-qualified name such as ``"femtoolkit.mesh"`` to obtain a
            child logger for a specific subsystem.

    Returns:
        A :class:`logging.Logger` configured with a
        :class:`logging.NullHandler`, so it stays silent until the calling
        application attaches its own handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(DEFAULT_LOG_LEVEL)
    return logger


_package_logger = logging.getLogger(PACKAGE_NAME)
_package_logger.addHandler(logging.NullHandler())
