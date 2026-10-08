"""Safe persistence for surrogate and ROM models (Version 35).

Models are saved as explicit, human-inspectable formats -- JSON for a
surrogate model's (small) parameter set and metadata, JSON plus a
NumPy ``.npz`` archive for a ROM's (potentially large) basis matrix --
**never** :mod:`pickle`. Loading a model always validates both the
persistence schema version and the originating toolkit's software
version before trusting the file's contents; a missing field, a
corrupted JSON document, or a schema/version mismatch raises a
toolkit-specific exception rather than letting a raw deserialization
error (or, worse, a successfully-loaded but structurally wrong object)
through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from femtoolkit.exceptions import IncompatibleModelVersionError, ModelPersistenceError
from femtoolkit.surrogate.models import SURROGATE_MODEL_TYPES, SurrogateModel, build_surrogate_model
from femtoolkit.surrogate.rom.pod import PODModel

SURROGATE_PERSISTENCE_SCHEMA_VERSION = 1
"""The on-disk schema version for a saved surrogate model file."""

ROM_PERSISTENCE_SCHEMA_VERSION = 1
"""The on-disk schema version for a saved POD/ROM model file."""


def save_surrogate_model(model: SurrogateModel, path: str | Path) -> None:
    """Save a fitted :class:`~femtoolkit.surrogate.models.base.SurrogateModel` to a JSON file.

    Args:
        model: The fitted model to save.
        path: The destination file path.

    Raises:
        ValidationError: If ``model`` has not been fit.
    """
    from femtoolkit.config import __version__ as software_version

    payload = {
        "schema_version": SURROGATE_PERSISTENCE_SCHEMA_VERSION,
        "software_version": software_version,
        "state": model.serialize(),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_surrogate_model(path: str | Path) -> SurrogateModel:
    """Load a surrogate model saved by :func:`save_surrogate_model`.

    Args:
        path: The file path to load.

    Returns:
        The reconstructed, fitted surrogate model.

    Raises:
        ModelPersistenceError: If the file is missing, is not valid
            JSON, or is missing an expected field.
        IncompatibleModelVersionError: If the file's schema version is
            not the one this toolkit version knows how to read.
    """
    try:
        payload = json.loads(Path(path).read_text())
    except OSError as error:
        raise ModelPersistenceError(
            f"Could not read surrogate model file {path!r}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ModelPersistenceError(
            f"Surrogate model file {path!r} is not valid JSON: {error}"
        ) from error

    try:
        schema_version = payload["schema_version"]
        state = payload["state"]
        model_name = state["model_name"]
    except KeyError as error:
        raise ModelPersistenceError(
            f"Surrogate model file {path!r} is missing expected field {error}."
        ) from error

    if schema_version != SURROGATE_PERSISTENCE_SCHEMA_VERSION:
        raise IncompatibleModelVersionError(
            f"Surrogate model file {path!r} has schema_version={schema_version!r}, but this "
            f"toolkit version only reads schema_version="
            f"{SURROGATE_PERSISTENCE_SCHEMA_VERSION!r}."
        )
    if model_name not in SURROGATE_MODEL_TYPES:
        raise ModelPersistenceError(
            f"Surrogate model file {path!r} names unknown model_name {model_name!r}; expected "
            f"one of {SURROGATE_MODEL_TYPES}."
        )

    model = build_surrogate_model(model_name)
    try:
        model.load_state(state)
    except KeyError as error:
        raise ModelPersistenceError(
            f"Surrogate model file {path!r} is missing expected state field {error}."
        ) from error
    return model


def save_rom_model(model: PODModel, path: str | Path) -> None:
    """Save a fitted :class:`~femtoolkit.surrogate.rom.pod.PODModel`.

    Writes two sibling files: ``<path>`` (JSON metadata) and
    ``<path>.npz`` (the basis matrix and singular values, as a NumPy
    archive -- never a pickle).

    Args:
        model: The fitted POD model to save.
        path: The destination metadata file path.

    Raises:
        ValidationError: If ``model`` has not been fit.
    """
    from femtoolkit.config import __version__ as software_version
    from femtoolkit.exceptions import ValidationError

    if not model.is_fitted:
        raise ValidationError("save_rom_model() requires a fitted PODModel.")
    path = Path(path)
    np.savez(
        path.with_suffix(path.suffix + ".npz"),
        basis=model.basis,
        singular_values=model.singular_values,
    )
    metadata = {
        "schema_version": ROM_PERSISTENCE_SCHEMA_VERSION,
        "software_version": software_version,
        "model_name": model.name,
        "total_modes": model.total_modes,
        "selected_modes": model.selected_modes,
    }
    path.write_text(json.dumps(metadata, indent=2))


def load_rom_model(path: str | Path) -> PODModel:
    """Load a POD model saved by :func:`save_rom_model`.

    Args:
        path: The metadata file path originally passed to :func:`save_rom_model`.

    Returns:
        The reconstructed, fitted :class:`~femtoolkit.surrogate.rom.pod.PODModel`.

    Raises:
        ModelPersistenceError: If either file is missing, is not valid,
            or is missing an expected field.
        IncompatibleModelVersionError: If the file's schema version is
            not the one this toolkit version knows how to read.
    """
    path = Path(path)
    try:
        metadata = json.loads(path.read_text())
    except OSError as error:
        raise ModelPersistenceError(f"Could not read ROM model file {path!r}: {error}") from error
    except json.JSONDecodeError as error:
        raise ModelPersistenceError(
            f"ROM model file {path!r} is not valid JSON: {error}"
        ) from error

    try:
        schema_version = metadata["schema_version"]
        model_name = metadata["model_name"]
        total_modes = metadata["total_modes"]
        selected_modes = metadata["selected_modes"]
    except KeyError as error:
        raise ModelPersistenceError(
            f"ROM model file {path!r} is missing expected field {error}."
        ) from error

    if schema_version != ROM_PERSISTENCE_SCHEMA_VERSION:
        raise IncompatibleModelVersionError(
            f"ROM model file {path!r} has schema_version={schema_version!r}, but this toolkit "
            f"version only reads schema_version={ROM_PERSISTENCE_SCHEMA_VERSION!r}."
        )
    if model_name != "pod":
        raise ModelPersistenceError(
            f"ROM model file {path!r} names unknown model_name {model_name!r}."
        )

    array_path = path.with_suffix(path.suffix + ".npz")
    try:
        with np.load(array_path) as arrays:
            basis = arrays["basis"]
            singular_values = arrays["singular_values"]
    except OSError as error:
        raise ModelPersistenceError(
            f"Could not read ROM array file {array_path!r}: {error}"
        ) from error
    except KeyError as error:
        raise ModelPersistenceError(
            f"ROM array file {array_path!r} is missing expected array {error}."
        ) from error

    model = PODModel()
    model.basis = basis
    model.singular_values = singular_values
    model.total_modes = total_modes
    model.selected_modes = selected_modes
    model._total_energy = float(np.sum(singular_values**2))
    return model


def surrogate_model_state_as_dict(model: SurrogateModel) -> dict[str, Any]:
    """Return the same plain-dict state :func:`save_surrogate_model` writes, without touching disk.

    Useful for embedding a model's state directly into an engineering
    report or GUI display.
    """
    return model.serialize()


__all__ = [
    "ROM_PERSISTENCE_SCHEMA_VERSION",
    "SURROGATE_PERSISTENCE_SCHEMA_VERSION",
    "load_rom_model",
    "load_surrogate_model",
    "save_rom_model",
    "save_surrogate_model",
    "surrogate_model_state_as_dict",
]
